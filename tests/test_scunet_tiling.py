import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from SCUNet_ONNX_Denoise import (
    TILE_PAD,
    _run_onnx_inference,
    _run_tiled,
    _create_windows_inference_session,
    _create_inference_session,
    _ensure_runtime_dependencies,
    _parse_args,
    _get_openvino_gpu_device,
    _select_windows_provider,
    _tile_starts,
    _tile_weight,
)


def _legacy_starts(length, tile_size):
    stride = tile_size * 3 // 4
    starts = list(range(0, max(1, length - tile_size), stride))
    if not starts or starts[-1] + tile_size < length:
        starts.append(max(0, length - tile_size))
    return starts


def _legacy_weight(height, width):
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    wx = np.maximum(np.minimum(x, 1.0 - x) * 2.0, 0.1)
    wy = np.maximum(np.minimum(y, 1.0 - y) * 2.0, 0.1)
    return wy[:, np.newaxis] * wx[np.newaxis, :]


class _IdentitySession:
    def __init__(self):
        self.input_shapes = []

    def run(self, output_names, inputs):
        tile = inputs["tile"]
        self.input_shapes.append(tile.shape)
        return [tile.copy()]


class _UnicodeErrorSession:
    def __init__(self, providers):
        self.providers = providers

    def get_providers(self):
        return self.providers

    def run(self, output_names, inputs):
        raw_message = "887A0006 Le GPU ne répondra pas".encode("cp1252")
        offset = raw_message.index(b"\xe9")
        raise UnicodeDecodeError(
            "utf-8", raw_message, offset, offset + 1, "invalid continuation byte"
        )


class _OpenVinoSessionOptions:
    def __init__(self):
        self.execution_mode = None
        self.enable_mem_pattern = True
        self.config = {}
        self.devices = []

    def add_session_config_entry(self, key, value):
        self.config[key] = value

    def add_provider_for_devices(self, devices, options):
        self.devices.append((devices, options))


class _OpenVinoOrt:
    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"

    def __init__(self, gpu_available=True):
        self.gpu_available = gpu_available
        self.options = None
        self.registered = []

    def register_execution_provider_library(self, name, path):
        self.registered.append((name, path))

    def get_ep_devices(self):
        devices = [
            SimpleNamespace(
                ep_name="OpenVINOExecutionProvider",
                ep_metadata={
                    "ov_device": "GPU",
                    "FULL_DEVICE_NAME": "Intel UHD Graphics",
                },
            )
        ] if self.gpu_available else []
        return devices

    def SessionOptions(self):
        self.options = _OpenVinoSessionOptions()
        return self.options

    def InferenceSession(self, model_path, sess_options):
        return SimpleNamespace(
            get_providers=lambda: [
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]
        )


class ScunetTilingTests(unittest.TestCase):
    def test_selected_tile_geometry_keeps_core_overlap_and_context(self):
        tile_size = 128
        starts = _tile_starts(224, tile_size)
        self.assertEqual(TILE_PAD, 128)
        self.assertEqual(starts, [0, 96])
        self.assertEqual(tile_size - (starts[1] - starts[0]), 32)
        self.assertEqual(tile_size + 2 * TILE_PAD, 384)

    def test_tile_starts_match_legacy_overlap_and_edge_alignment(self):
        for length in (1, 128, 257, 1024, 3072):
            for tile_size in (128, 256, 384, 512):
                with self.subTest(length=length, tile_size=tile_size):
                    self.assertEqual(
                        _tile_starts(length, tile_size),
                        _legacy_starts(length, tile_size),
                    )

    def test_tile_weights_match_legacy_blend_mask(self):
        for height, width in ((1, 1), (64, 128), (256, 256)):
            with self.subTest(height=height, width=width):
                np.testing.assert_array_equal(
                    _tile_weight(np, height, width),
                    _legacy_weight(height, width),
                )

    def test_fixed_shape_tiles_merge_to_identity_without_seams(self):
        tile_size = 128
        model_size = tile_size + 2 * TILE_PAD
        image = np.random.default_rng(42).random((3, 224, 224), dtype=np.float32)
        session = _IdentitySession()

        result, timing = _run_tiled(
            np,
            session,
            "tile",
            "output",
            np.float32,
            image,
            tile_size,
            False,
            model_height=model_size,
            model_width=model_size,
        )

        self.assertEqual(timing["tiles"], 4)
        self.assertEqual(model_size, 384)
        self.assertEqual(set(session.input_shapes), {(1, 3, 384, 384)})
        np.testing.assert_allclose(result, image, rtol=0.0, atol=2e-6)

    def test_directml_native_decode_error_reports_tile_and_native_message(self):
        session = _UnicodeErrorSession(
            ["DmlExecutionProvider", "CPUExecutionProvider"]
        )
        with self.assertRaisesRegex(
            RuntimeError, r"tile 2/5 at y=256, x=128"
        ) as raised:
            _run_onnx_inference(
                session,
                "output",
                "input",
                np.zeros((1, 3, 384, 384), dtype=np.float32),
                2,
                5,
                256,
                128,
            )
        self.assertIn("Le GPU ne répondra pas", str(raised.exception))
        self.assertIn("CPU provider fallback is disabled", str(raised.exception))
        self.assertIn("DXGI_ERROR_DEVICE_HUNG", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, UnicodeDecodeError)

    def test_non_directml_decode_error_is_not_rewritten(self):
        session = _UnicodeErrorSession(["CPUExecutionProvider"])
        with self.assertRaises(UnicodeDecodeError):
            _run_onnx_inference(
                session,
                "output",
                "input",
                np.zeros((1, 3, 384, 384), dtype=np.float32),
                1,
                1,
                0,
                0,
            )

    def test_openvino_option_installs_only_plugin(self):
        class FakeSiril:
            def __init__(self):
                self.installs = []

            def ensure_installed(self, *packages):
                self.installs.append(packages)

        siril = FakeSiril()
        _ensure_runtime_dependencies(siril, "openvino")
        self.assertEqual(
            siril.installs, [("numpy", "onnxruntime-ep-openvino")]
        )
        self.assertEqual(_parse_args([]).provider, "directml")
        self.assertEqual(_parse_args(["--provider", "openvino"]).provider, "openvino")

    def test_openvino_session_selects_gpu_and_disables_cpu_fallback(self):
        fake_ort = _OpenVinoOrt()
        plugin = SimpleNamespace(
            get_library_path=lambda: "openvino-plugin.dll",
            get_ep_name=lambda: "OpenVINOExecutionProvider",
        )
        previous_registration = __import__(
            "SCUNet_ONNX_Denoise"
        )._OPENVINO_REGISTERED_LIBRARY
        module = __import__("SCUNet_ONNX_Denoise")
        try:
            module._OPENVINO_REGISTERED_LIBRARY = None
            with mock.patch.dict(
                sys.modules, {"onnxruntime_ep_openvino": plugin}
            ):
                session = _create_inference_session(
                    fake_ort, "model.onnx", "openvino"
                )
            self.assertIn(
                "OpenVINOExecutionProvider", session.get_providers()
            )
            self.assertEqual(
                fake_ort.registered,
                [("scunet_openvino", "openvino-plugin.dll")],
            )
            self.assertEqual(
                fake_ort.options.config[
                    "session.disable_cpu_ep_fallback"
                ],
                "1",
            )
            self.assertEqual(fake_ort.options.devices[0][0][0].ep_metadata["ov_device"], "GPU")
            self.assertEqual(
                module._OPENVINO_SELECTED_DEVICE, "Intel UHD Graphics"
            )
        finally:
            module._OPENVINO_REGISTERED_LIBRARY = previous_registration

    def test_openvino_session_refuses_cpu_only_device(self):
        fake_ort = _OpenVinoOrt(gpu_available=False)
        plugin = SimpleNamespace(
            get_library_path=lambda: "openvino-plugin.dll",
            get_ep_name=lambda: "OpenVINOExecutionProvider",
        )
        module = __import__("SCUNet_ONNX_Denoise")
        previous_registration = module._OPENVINO_REGISTERED_LIBRARY
        try:
            module._OPENVINO_REGISTERED_LIBRARY = None
            with mock.patch.dict(
                sys.modules, {"onnxruntime_ep_openvino": plugin}
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "refusing to fall back to CPU"
                ):
                    _create_inference_session(
                        fake_ort, "model.onnx", "openvino"
                    )
        finally:
            module._OPENVINO_REGISTERED_LIBRARY = previous_registration

    def test_openvino_gpu_detection_and_automatic_selection(self):
        plugin = SimpleNamespace(
            get_library_path=lambda: "openvino-plugin.dll",
            get_ep_name=lambda: "OpenVINOExecutionProvider",
        )
        module = __import__("SCUNet_ONNX_Denoise")
        previous_registration = module._OPENVINO_REGISTERED_LIBRARY
        try:
            with mock.patch.dict(
                sys.modules, {"onnxruntime_ep_openvino": plugin}
            ):
                gpu = _get_openvino_gpu_device(_OpenVinoOrt())
                no_gpu = _get_openvino_gpu_device(
                    _OpenVinoOrt(gpu_available=False)
                )
            self.assertEqual(gpu.ep_metadata["ov_device"], "GPU")
            self.assertIsNone(no_gpu)
            self.assertEqual(
                _select_windows_provider("auto", gpu is not None), "openvino"
            )
            self.assertEqual(
                _select_windows_provider("auto", no_gpu is not None),
                "directml",
            )
            self.assertEqual(
                _select_windows_provider("directml", True), "directml"
            )
            self.assertEqual(
                _select_windows_provider("openvino", False), "openvino"
            )
        finally:
            module._OPENVINO_REGISTERED_LIBRARY = previous_registration

    def test_automatic_openvino_activation_failure_retries_directml(self):
        openvino_error = RuntimeError("model unsupported by OpenVINO")
        session = object()
        with mock.patch(
            "SCUNet_ONNX_Denoise._create_inference_session",
            side_effect=[openvino_error, session],
        ) as create_session:
            actual_session, provider, fallback_error = (
                _create_windows_inference_session(
                    object(), "model.onnx", "auto", True
                )
            )
        self.assertIs(actual_session, session)
        self.assertEqual(provider, "directml")
        self.assertIs(fallback_error, openvino_error)
        self.assertEqual(
            [call.args[2] for call in create_session.call_args_list],
            ["openvino", "directml"],
        )

    def test_manual_openvino_activation_failure_does_not_fallback(self):
        openvino_error = RuntimeError("model unsupported by OpenVINO")
        with mock.patch(
            "SCUNet_ONNX_Denoise._create_inference_session",
            side_effect=openvino_error,
        ) as create_session:
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                _create_windows_inference_session(
                    object(), "model.onnx", "openvino", True
                )
        create_session.assert_called_once()

    def test_gui_openvino_load_installs_through_sirilpy_module(self):
        import SCUNet_DenoiseV2 as gui

        siril_interface = SimpleNamespace(
            log=mock.Mock(),
        )
        session = SimpleNamespace(
            get_inputs=lambda: [
                SimpleNamespace(shape=[1, 3, 384, 384])
            ]
        )
        with (
            mock.patch.object(
                gui._onnx_tools, "_ensure_runtime_dependencies"
            ) as ensure_dependencies,
            mock.patch.object(
                gui._onnx_tools,
                "_create_windows_inference_session",
                return_value=(session, "openvino", None),
            ),
            mock.patch.object(gui.os.path, "isfile", return_value=True),
        ):
            loaded, core_size, provider = gui._load_onnx_model(
                siril_interface,
                "scunet.pth",
                "Auto",
                provider="openvino",
            )

        ensure_dependencies.assert_called_once_with(gui.s, "openvino")
        self.assertIs(loaded, session)
        self.assertEqual(core_size, 128)
        self.assertEqual(provider, "openvino")


if __name__ == "__main__":
    unittest.main()
