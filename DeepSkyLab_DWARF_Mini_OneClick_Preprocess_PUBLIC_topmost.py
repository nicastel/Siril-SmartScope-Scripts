# Deep SkyLab - DWARF Mini One-Click Preprocess for Siril
# -----------------------------------------------------------------------------
# What this script does (in plain English):
#
#  1) Reads your DWARF session metadata (shotsInfo.json) to learn exposure, gain,
#     binning, and which IR/filter mode you used.
#  2) Finds the best-matching calibration frames in CALI_FRAME (bias/dark/flat).
#  3) Copies everything into a fresh working folder (so your originals stay clean).
#  4) Builds master calibration frames, calibrates your light subs, registers them,
#     optionally culls the worst frames by FWHM, then stacks to a final *linear* FITS.
#
# Folder expectations (DWARF export layout):
#
#  Astronomy/                       <-- you should run this on a COPY of this folder
#    CALI_FRAME/
#      bias/<cam_*...>/*.fits
#      dark/cam_0/*.fits            (DWARF Mini dark library is organized by camera)
#      flat/<cam_*...>/*.fits
#    DWARF_RAW_.../
#      shotsInfo.json
#      *.fits                       (light subs + some DWARF-made extras we ignore)
#      thumbnails/                  (ignored - not scanned)
#
# Outputs:
#
#  - Working folder (created next to your lights):
#      <DWARF_RAW_...>/DSL_SIRIL_PROCESS*/
#
#  - Final *linear* stack (saved at the Astronomy root):
#      <Astronomy>/RESTACKED/DSL_STACK_... .fits
#
#  - Per-frame stats CSV (handy for nerdy comparisons):
#      <DWARF_RAW_...>/DSL_SIRIL_PROCESS*/DSL_seqstat_r_pp_light.csv
#
# Notes for humans:
#  - cam_0 = TELE  |  cam_1 = WIDE (based on DWARF folder naming).
#  - "IR" in shotsInfo.json is used to pick the best flat folder (Astro vs Dual-band).
#  - FWHM filtering keeps the *tightest* stars (lower FWHM = better).
#
# Requirements:
#  - Siril 1.4+ with Python scripting enabled (sirilpy).
#
# If you found this file because something broke: scroll up in the Siril console.
# The line right before an error usually tells you which command was unhappy.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sirilpy as s


# --------------------------- constants (names used throughout the script) --------------------

SCRIPT_NAME = "Deep SkyLab — DWARF Mini one-click preprocess"
SCRIPT_VERSION = "0.7.7"

PROCESS_DIR_BASE = "DSL_SIRIL_PROCESS"
RESTACKED_DIR_NAME = "RESTACKED"


# --------------------------- path helpers (Siril likes forward slashes) ----------------------

def _posix(p: Union[Path, str]) -> str:
    return str(p).replace("\\", "/")


def _quote_if_needed(tok: str) -> str:
    # For Siril commands where the path is a positional argument (cd/load),
    # quoting is safe and recommended when spaces exist.
    return f"\"{tok}\"" if (" " in tok or "\t" in tok) else tok


def _cd_arg(p: Path) -> str:
    return _quote_if_needed(_posix(p))


def _rel(from_dir: Path, to_path: Path) -> str:
    """Relative path for Siril option values (-out=, -bias=, -dark=, -flat=).

    IMPORTANT: Siril option parsing can choke on quoted paths with spaces.
    Using relative paths like ../masterbias avoids that.
    """
    rel = os.path.relpath(str(to_path), start=str(from_dir))
    return rel.replace("\\", "/")


# --------------------------- filesystem helpers (safe, boring file ops) ---------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rmtree_retry(path: Path, tries: int = 5, delay_s: float = 0.5) -> None:
    """Best-effort recursive delete that handles transient Windows locks."""

    def _onerror(func, p, exc_info):
        # Try to clear read-only bit, then retry.
        try:
            os.chmod(p, 0o777)
        except Exception:
            pass
        try:
            func(p)
        except Exception:
            pass

    last_exc: Optional[Exception] = None
    for _ in range(max(1, tries)):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except Exception as e:
            last_exc = e
            time.sleep(delay_s)

    if last_exc:
        raise last_exc


def _next_available_dir(parent: Path, base: str) -> Path:
    """Return parent/base if free, else parent/base_01, base_02, ..."""
    p0 = parent / base
    if not p0.exists():
        return p0
    for i in range(1, 100):
        cand = parent / f"{base}_{i:02d}"
        if not cand.exists():
            return cand
    return parent / f"{base}_{int(time.time())}"


def _glob_fits(folder: Path) -> List[Path]:
    exts = ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS")
    out: List[Path] = []
    for pat in exts:
        out.extend(folder.glob(pat))
    out = [p for p in out if p.is_file()]
    return sorted(set(out))


def _sanitize_filename(s0: str) -> str:
    s0 = (s0 or "").strip()
    s0 = re.sub(r'[<>:"/\\|?*]', "_", s0)
    s0 = re.sub(r"\s+", "_", s0)
    return s0


def _copy_renamed(src: List[Path], dst_dir: Path, prefix: str, digits: int = 5) -> List[Path]:
    """Copy files into dst_dir, renaming them to prefix_00001.fits (no spaces)."""
    _safe_mkdir(dst_dir)
    out: List[Path] = []
    for i, f in enumerate(src, start=1):
        dst = dst_dir / f"{prefix}_{i:0{digits}d}.fits"
        shutil.copy2(f, dst)
        out.append(dst)
    return out


# --------------------------- DWARF shotsInfo.json (session metadata) ------------------------

@dataclass
class ShotsInfo:
    target: str
    exp_s: float
    gain: int
    ir: str
    binning: int
    min_temp: Optional[int]
    max_temp: Optional[int]
    shots_taken: Optional[int]
    shots_stacked: Optional[int]

    @property
    def mean_temp(self) -> Optional[float]:
        if self.min_temp is None or self.max_temp is None:
            return None
        return (self.min_temp + self.max_temp) / 2.0


def _read_shotsinfo(shotsinfo_path: Path) -> ShotsInfo:
    with shotsinfo_path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    target = str(d.get("target", "UNKNOWN"))
    exp_s = float(d.get("exp", 0))
    gain = int(d.get("gain", 0))
    ir = str(d.get("ir", "UNKNOWN"))

    binning_raw = str(d.get("binning", "1*1"))
    try:
        binning = int(binning_raw.split("*")[0])
    except Exception:
        binning = 1

    min_temp = d.get("minTemp", None)
    max_temp = d.get("maxTemp", None)
    min_temp = int(min_temp) if min_temp is not None else None
    max_temp = int(max_temp) if max_temp is not None else None

    shots_taken = d.get("shotsTaken", None)
    shots_stacked = d.get("shotsStacked", None)
    shots_taken = int(shots_taken) if shots_taken is not None else None
    shots_stacked = int(shots_stacked) if shots_stacked is not None else None

    return ShotsInfo(
        target=target,
        exp_s=exp_s,
        gain=gain,
        ir=ir,
        binning=binning,
        min_temp=min_temp,
        max_temp=max_temp,
        shots_taken=shots_taken,
        shots_stacked=shots_stacked,
    )


def _detect_cam_name(folder_name: str) -> str:
    n = folder_name.upper()
    if "TELE" in n:
        return "cam_0"
    if "WIDE" in n:
        return "cam_1"
    return "cam_0"


def _detect_ir_code(ir_str: str) -> Optional[int]:
    s0 = (ir_str or "").strip().lower()
    if not s0:
        return None
    if "astro" in s0:
        return 1
    if "dual" in s0 or "duo" in s0 or "band" in s0 or "narrow" in s0:
        return 2
    if "none" in s0 or "off" in s0 or "clear" in s0 or "ircut" in s0:
        return 0
    return None


def _pick_best_calib_subfolder(parent: Path, cam_name: str, ir_code: Optional[int], gain: int) -> Optional[Path]:
    """Pick best matching subfolder in CALI_FRAME/{bias|flat}."""
    if not parent.is_dir():
        return None

    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.lower().startswith(cam_name.lower())]
    if not candidates:
        return None

    def score(p: Path) -> int:
        name = p.name.lower()
        sc = 0
        if name == cam_name.lower():
            sc += 5
        if ir_code is not None:
            if f"ir_{ir_code}" in name:
                sc += 10
            elif "ir_" in name:
                sc -= 2
        if f"gain_{gain}" in name:
            sc += 3
        elif "gain_" in name:
            sc -= 1
        # prefer slightly more specific folders
        sc += len(name) // 10
        return sc

    return sorted(candidates, key=score, reverse=True)[0]


# --------------------------- dark selection (match exposure/gain/bin/temp) ------------------

@dataclass
class DarkMeta:
    exp_s: float
    gain: int
    binning: int
    temp_c: int


_DARK_RE = re.compile(
    r"dark_exp_(?P<exp>[0-9]+\.?[0-9]*)_gain_(?P<gain>[0-9]+)_bin_(?P<bin>[0-9]+)_(?P<temp>[0-9]+)C",
    re.IGNORECASE,
)


def _parse_dark_filename(name: str) -> Optional[DarkMeta]:
    m = _DARK_RE.search(name)
    if not m:
        return None
    try:
        return DarkMeta(
            exp_s=float(m.group("exp")),
            gain=int(m.group("gain")),
            binning=int(m.group("bin")),
            temp_c=int(m.group("temp")),
        )
    except Exception:
        return None


def _select_matching_darks(dark_dir: Path, shots: ShotsInfo) -> List[Path]:
    files = _glob_fits(dark_dir)
    if not files:
        return []

    exp_tol = max(0.05, shots.exp_s * 0.02)  # DWARF uses odd decimals sometimes

    candidates: List[Tuple[Path, DarkMeta]] = []
    for f in files:
        meta = _parse_dark_filename(f.name)
        if not meta:
            continue
        if meta.gain != shots.gain:
            continue
        if meta.binning != shots.binning:
            continue
        if abs(meta.exp_s - shots.exp_s) > exp_tol:
            continue
        candidates.append((f, meta))

    if not candidates:
        return []

    # Prefer temps inside session range
    # Tiny bonus: matching dark temperature matters more than most people think (until it *really* does).
    if shots.min_temp is not None and shots.max_temp is not None:
        in_range = [f for (f, m) in candidates if shots.min_temp <= m.temp_c <= shots.max_temp]
        if in_range:
            return sorted(in_range)

    # Else closest to mean temp (or median)
    temps = [m.temp_c for (_, m) in candidates]
    target_t = shots.mean_temp if shots.mean_temp is not None else sorted(temps)[len(temps) // 2]
    best_dist = min(abs(m.temp_c - target_t) for (_, m) in candidates)
    chosen = [f for (f, m) in candidates if abs(m.temp_c - target_t) == best_dist]
    return sorted(chosen)


# --------------------------- FITS layer check (CFA 1-layer vs RGB 3-layer) ------------------

_TEMP_SUFFIX_RE = re.compile(r".*_[+-]?\d+C\.(fit|fits|fts)$", re.IGNORECASE)


def _fits_layer_count(path: Path) -> Optional[int]:
    """Cheap FITS header peek to estimate # layers.

    - NAXIS<=2 -> 1 layer
    - NAXIS=3 and NAXIS3=3 -> 3 layers

    Returns None if it can't parse.
    """
    try:
        header_cards: List[str] = []
        with path.open("rb") as f:
            for _ in range(20):
                block = f.read(2880)
                if not block:
                    break
                for i in range(0, len(block), 80):
                    card = block[i : i + 80].decode("ascii", errors="ignore")
                    header_cards.append(card)
                    if card.startswith("END"):
                        raise StopIteration
    except StopIteration:
        pass
    except Exception:
        return None

    kv: Dict[str, str] = {}
    for c in header_cards:
        if "=" in c[:10]:
            key = c[:8].strip()
            val = c.split("=", 1)[1].split("/", 1)[0].strip()
            kv[key] = val

    try:
        naxis = int(kv.get("NAXIS", "2"))
    except Exception:
        return None

    if naxis <= 2:
        return 1

    try:
        naxis3 = int(kv.get("NAXIS3", "1"))
    except Exception:
        naxis3 = 1

    return naxis3


def _select_light_files(target_dir: Path) -> Tuple[List[Path], Dict[int, int], List[Path]]:
    """Return (selected_subs, layer_hist, excluded_fits).

    Excludes DWARF products like stacked*.fits and filters by majority layer count
    to prevent Siril sequence aborts.
    """
    allfits = _glob_fits(target_dir)

    excluded: List[Path] = []

    # Exclude obvious non-subs
    nonstack: List[Path] = []
    for p in allfits:
        n = p.name.lower()
        if "stacked" in n:
            excluded.append(p)
            continue
        if n.startswith("pp_") or n.startswith("r_") or n.startswith("dsl_"):
            excluded.append(p)
            continue
        nonstack.append(p)

    # Prefer classic DWARF raw-sub naming: ..._27C.fits
    temp_named = [p for p in nonstack if _TEMP_SUFFIX_RE.match(p.name)]
    candidates = temp_named if len(temp_named) >= max(5, len(nonstack) // 2) else nonstack

    # Layer-count majority filter
    layers: Dict[Path, Optional[int]] = {p: _fits_layer_count(p) for p in candidates}
    hist: Dict[int, int] = {}
    for _, n in layers.items():
        if n is None:
            continue
        hist[n] = hist.get(n, 0) + 1

    if hist:
        majority = sorted(hist.items(), key=lambda kv: kv[1], reverse=True)[0][0]
        selected = [p for p in candidates if layers.get(p, None) == majority]
        # Any candidate with a different layer count is excluded
        for p in candidates:
            if layers.get(p, None) != majority:
                excluded.append(p)
    else:
        selected = candidates

    return sorted(selected), hist, sorted(set(excluded))


# --------------------------- tiny GUI (pick target + keep %) --------------------------------

def _tk_choose_astronomy_dir() -> Optional[Path]:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    #root.withdraw()

    # IMPORTANT: On Windows, Siril can steal focus and make dialogs look like they "didn't open".
    # For sanity, we force the file picker to stay above Siril (and basically everything else).
    #root.attributes("-topmost", True)
    #root.lift()
    #root.focus_force()
    #root.update()

    d = filedialog.askdirectory(
        parent=root,
        title="Select the copied DWARF 'Astronomy' folder (work on a copy)",
    )
    root.destroy()
    return Path(d) if d else None


def _tk_select_target(astronomy_dir: Path) -> Tuple[Optional[Path], Optional[int], bool]:
    """Returns: (selected_target_dir, keep_percent, overwrite_process)."""
    import tkinter as tk
    from tkinter import ttk, messagebox

    targets = [p for p in astronomy_dir.iterdir() if p.is_dir() and p.name.startswith("DWARF_RAW_")]
    targets = sorted(targets, key=lambda p: p.name)

    if not targets:
        # Force the error box above Siril so it doesn't hide behind the main app window.
        tmp = tk.Tk()
        tmp.withdraw()
        tmp.attributes("-topmost", True)
        tmp.lift()
        tmp.focus_force()
        tmp.update()
        messagebox.showerror(SCRIPT_NAME, "No DWARF_RAW_* folders found in that Astronomy folder.", parent=tmp)
        tmp.destroy()
        return None, None, False

    rows: List[str] = []
    for t in targets:
        cam = _detect_cam_name(t.name).replace("cam_", "CAM_")
        shots_path = t / "shotsInfo.json"
        if shots_path.exists():
            try:
                sh = _read_shotsinfo(shots_path)
                subs, _, _ = _select_light_files(t)
                rows.append(
                    f"{cam:5s} | {sh.target:20s} | exp {sh.exp_s:g}s | gain {sh.gain:3d} | IR {sh.ir:8s} | {len(subs):4d} subs | {t.name}"
                )
            except Exception:
                rows.append(f"{cam:5s} | (shotsInfo unreadable) | {t.name}")
        else:
            rows.append(f"{cam:5s} | (no shotsInfo.json) | {t.name}")

    win = tk.Tk()
    win.title(f"{SCRIPT_NAME} (Siril) — v{SCRIPT_VERSION}")

    # This window is the #1 place people "lose" the script, because it likes to open behind Siril.
    # Making it always-on-top keeps it visible on multi-monitor setups and when Siril steals focus.
    win.attributes("-topmost", True)
    win.lift()
    win.focus_force()
    win.after(150, lambda: (win.lift(), win.focus_force()))

    frm = ttk.Frame(win, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")

    ttk.Label(frm, text="Select the DWARF_RAW_* folder to process:").grid(row=0, column=0, sticky="w")

    lb = tk.Listbox(frm, width=150, height=min(18, max(8, len(rows))))
    for r in rows:
        lb.insert(tk.END, r)
    lb.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(6, 8))
    lb.selection_set(0)

    ttk.Label(frm, text="Keep best frames (%):").grid(row=2, column=0, sticky="w")
    keep_var = tk.IntVar(value=95)
    keep_slider = ttk.Scale(frm, from_=50, to=100, orient="horizontal", command=lambda v: keep_var.set(int(float(v))))
    keep_slider.set(95)
    keep_slider.grid(row=2, column=1, sticky="ew", padx=(10, 10))
    ttk.Label(frm, textvariable=keep_var, width=4).grid(row=2, column=2, sticky="w")

    overwrite_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text=f"Overwrite existing {PROCESS_DIR_BASE} folder", variable=overwrite_var).grid(
        row=3, column=0, columnspan=3, sticky="w", pady=(8, 0)
    )

    result = {"ok": False}

    def on_run() -> None:
        sel = lb.curselection()
        if not sel:
            messagebox.showerror(SCRIPT_NAME, "Select a target folder first.", parent=win)
            return
        result["ok"] = True
        result["idx"] = int(sel[0])
        win.destroy()

    def on_cancel() -> None:
        result["ok"] = False
        win.destroy()

    btns = ttk.Frame(frm)
    btns.grid(row=4, column=0, columnspan=3, sticky="e", pady=(10, 0))
    ttk.Button(btns, text="Cancel", command=on_cancel).grid(row=0, column=0, padx=(0, 10))
    ttk.Button(btns, text="Run", command=on_run).grid(row=0, column=1)

    frm.columnconfigure(1, weight=1)
    win.columnconfigure(0, weight=1)
    win.rowconfigure(0, weight=1)

    win.mainloop()

    if not result.get("ok"):
        return None, None, False

    chosen = targets[result["idx"]]
    return chosen, int(keep_var.get()), bool(overwrite_var.get())


# --------------------------- Siril helper (small wrappers around Siril commands) ------------

def _save_single_as_master(siril: s.SirilInterface, src_path: Path, master_base: Path) -> None:
    """Load a single FITS (by full path) and save as master_base (no extension)."""
    siril.cmd("load", _cd_arg(src_path))
    # `save` writes into the *current* working directory. Using a relative path here avoids
    # quoting issues on Windows paths with spaces. (Siril parses some options… creatively.)
    out_rel = _rel(src_path.parent, master_base)
    siril.cmd("save", out_rel)


# --------------------------- main workflow (bias → dark/flat → lights) ----------------------

def main() -> None:
    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception as e:
        print(f"[{SCRIPT_NAME}] Could not connect to Siril: {e}")
        return

    # Version guard
    try:
        siril.cmd("requires", "1.4.0")
    except Exception:
        try:
            siril.error_messagebox("This script requires Siril 1.4 or newer (Python scripting).")
        except Exception:
            pass
        return

    siril.log(f"{SCRIPT_NAME} starting… (v{SCRIPT_VERSION})")

    astronomy_dir = _tk_choose_astronomy_dir()
    if astronomy_dir is None:
        siril.log("Cancelled (no Astronomy folder selected).")
        return

    cali_dir = astronomy_dir / "CALI_FRAME"
    if not cali_dir.is_dir():
        siril.error_messagebox("CALI_FRAME folder not found inside the selected Astronomy folder.")
        return

    target_dir, keep_percent, overwrite_process = _tk_select_target(astronomy_dir)
    if target_dir is None or keep_percent is None:
        siril.log("Cancelled (no target selected).")
        return

    shots_path = target_dir / "shotsInfo.json"
    if not shots_path.exists():
        siril.error_messagebox("shotsInfo.json not found in the selected target folder.")
        return

    shots = _read_shotsinfo(shots_path)
    cam_name = _detect_cam_name(target_dir.name)
    ir_code = _detect_ir_code(shots.ir)

    siril.log(f"Target folder: {target_dir}")
    siril.log(
        f"Detected: cam={cam_name}, target='{shots.target}', exp={shots.exp_s}s, gain={shots.gain}, IR='{shots.ir}', temps={shots.min_temp}-{shots.max_temp}C"
    )
    siril.log(f"Frame culling: keep best {keep_percent}% (by FWHM)")

    # ------------------ pick calibration subfolders ------------------

    bias_src = _pick_best_calib_subfolder(cali_dir / "bias", cam_name, ir_code, shots.gain)
    flat_src = _pick_best_calib_subfolder(cali_dir / "flat", cam_name, ir_code, shots.gain)
    dark_src = (cali_dir / "dark" / cam_name) if (cali_dir / "dark" / cam_name).is_dir() else None

    if bias_src is None:
        siril.error_messagebox(f"No bias folder found for {cam_name} in CALI_FRAME/bias.")
        return

    siril.log(f"Bias folder chosen: {bias_src}")
    if flat_src:
        siril.log(f"Flat folder chosen: {flat_src}")
    else:
        siril.log("WARNING: no flat folder matched. The script will run without flats (not recommended).")
    if dark_src:
        siril.log(f"Dark folder chosen: {dark_src}")
    else:
        siril.log("WARNING: no dark folder found for this camera. The script will run without darks.")

    # ------------------ select light files (ignore thumbnails/stacked/etc) -----

    light_files, layer_hist, excluded_fits = _select_light_files(target_dir)
    if not light_files:
        siril.error_messagebox("No suitable light subframes found in the target folder.")
        return

    if excluded_fits:
        # Log only a few to keep the console readable
        sample = ", ".join([p.name for p in excluded_fits[:6]])
        more = "" if len(excluded_fits) <= 6 else f" (+{len(excluded_fits) - 6} more)"
        siril.log(f"Excluded FITS (not subs): {sample}{more}")

    if layer_hist:
        siril.log("Light FITS layer histogram: " + ", ".join(f"{k}L={v}" for k, v in sorted(layer_hist.items())))
    else:
        siril.log("Light FITS layer histogram: (could not parse headers)")

    majority_layers = None
    if layer_hist:
        majority_layers = sorted(layer_hist.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    is_cfa = True if majority_layers is None else (majority_layers == 1)
    if majority_layers is None:
        siril.log("Lights layer count: unknown -> assuming CFA/RAW.")
    else:
        siril.log(f"Lights majority layer count: {majority_layers} -> treating as {'CFA/RAW' if is_cfa else 'RGB'} dataset.")

    # ------------------ prepare process directories ----------------------------

    # Always try to close any open image/sequence first.
    try:
        siril.cmd("close")
    except Exception:
        pass

    # Determine process_dir (safe overwrite)
    base_dir = target_dir / PROCESS_DIR_BASE
    process_dir = base_dir

    if base_dir.exists():
        if overwrite_process:
            siril.log(f"Overwriting existing {PROCESS_DIR_BASE} folder…")
            try:
                _rmtree_retry(base_dir, tries=6, delay_s=0.5)
            except Exception as e:
                # Fall back to a new unique folder
                process_dir = _next_available_dir(target_dir, PROCESS_DIR_BASE)
                siril.log(
                    f"WARNING: Could not delete existing {PROCESS_DIR_BASE} (likely Windows file lock).\n"
                    f"Reason: {e}\n"
                    f"Continuing with a new folder: {process_dir.name}"
                )
        else:
            process_dir = _next_available_dir(target_dir, PROCESS_DIR_BASE)
            siril.log(f"Keeping existing {PROCESS_DIR_BASE}; using new folder: {process_dir.name}")

    bias_work = process_dir / "CALIB_BIAS"
    dark_work = process_dir / "CALIB_DARK"
    flat_work = process_dir / "CALIB_FLAT"
    light_work = process_dir / "LIGHTS"

    _safe_mkdir(process_dir)

    # Copy & rename calibration frames (no spaces) into their work dirs
    bias_files = _glob_fits(bias_src)
    if not bias_files:
        siril.error_messagebox("No bias FITS files found in the chosen bias folder.")
        return
    _copy_renamed(bias_files, bias_work, "biasraw", digits=3)

    flat_files: List[Path] = []
    if flat_src is not None:
        flat_files = _glob_fits(flat_src)
        if flat_files:
            _copy_renamed(flat_files, flat_work, "flatraw", digits=3)

    dark_files: List[Path] = []
    if dark_src is not None:
        dark_files = _select_matching_darks(dark_src, shots)
        if dark_files:
            _copy_renamed(dark_files, dark_work, "darkraw", digits=3)

    # Copy & rename lights into work dir
    _copy_renamed(light_files, light_work, "lightraw", digits=5)

    siril.log(
        f"Copied into process folder: lights={len(light_files)}, bias={len(bias_files)}, flat={len(flat_files)}, dark={len(dark_files)}"
    )

    # ------------------ Siril preferences --------------------------------------

    siril.cmd("setext", "fits")
    siril.cmd("set32bits")

    # Masters live in process_dir root (no extension in the basename)
    master_bias = process_dir / "masterbias"
    master_dark = process_dir / "masterdark"
    master_flat = process_dir / "masterflat"

    # ------------------ build master bias --------------------------------------

    siril.log("Building master bias…")
    siril.cmd("cd", _cd_arg(bias_work))

    if len(bias_files) == 1:
        # Siril sometimes doesn't build a usable sequence from a single frame.
        # Treat the single bias as the master.
        one = _glob_fits(bias_work)[0]
        _save_single_as_master(siril, one, master_bias)
        siril.log("Master bias: single frame → saved")
    else:
        # CONVERT creates the *sequence* named 'bias' from all images in CWD.
        siril.cmd("convert", "bias")
        out_rel = _rel(bias_work, master_bias)
        siril.cmd("stack", "bias", "median", "-nonorm", f"-out={out_rel}")

    # ------------------ build master dark (optional) ---------------------------

    # DWARF often provides only a handful of darks. Siril cannot create a usable
    # sequence from a single frame, so we require >=2 frames to build a master.
    has_dark = len(dark_files) >= 2
    if len(dark_files) == 1:
        siril.log("WARNING: only 1 matching dark frame found — skipping master dark (need >=2).")

    if has_dark:
        siril.log("Building master dark (bias-subtracted)…")
        siril.cmd("cd", _cd_arg(dark_work))

        # Build dark sequence from all frames in this folder
        siril.cmd("convert", "dark")

        # IMPORTANT: subtract master bias from darks BEFORE stacking, otherwise
        # providing both -bias and -dark during light calibration would double-subtract bias.
        bias_rel = _rel(dark_work, master_bias)
        cal_dark: List[str] = ["calibrate", "dark", f"-bias={bias_rel}", "-prefix=pp_"]
        if is_cfa:
            cal_dark.append("-cfa")
        siril.cmd(*cal_dark)  # -> pp_dark

        out_rel = _rel(dark_work, master_dark)
        siril.cmd("stack", "pp_dark", "median", "-nonorm", f"-out={out_rel}")
    else:
        siril.log("Skipping master dark (no matching dark files found).")

    # ------------------ build master flat (optional) ---------------------------

    has_flat = len(flat_files) > 0
    if has_flat:
        siril.log("Building master flat…")
        siril.cmd("cd", _cd_arg(flat_work))
        siril.cmd("convert", "flat")

        # Calibrate flats with master bias
        bias_rel = _rel(flat_work, master_bias)
        cal_flat: List[str] = ["calibrate", "flat", f"-bias={bias_rel}", "-prefix=pp_"]
        if is_cfa:
            cal_flat.append("-cfa")
        siril.cmd(*cal_flat)  # -> pp_flat

        out_rel = _rel(flat_work, master_flat)
        siril.cmd("stack", "pp_flat", "median", "-norm=mul", f"-out={out_rel}")
    else:
        siril.log("Skipping master flat (no flat files found).")

    # ------------------ convert + calibrate + register + stack lights ----------

    siril.log("Converting lights to a Siril sequence…")
    siril.cmd("cd", _cd_arg(light_work))
    siril.cmd("convert", "light")

    siril.log("Calibrating lights…")
    calib_args: List[str] = [
        "calibrate",
        "light",
        f"-bias={_rel(light_work, master_bias)}",
        "-prefix=pp_",
    ]

    if is_cfa:
        calib_args += ["-cfa", "-debayer"]

    if has_flat:
        calib_args += [f"-flat={_rel(light_work, master_flat)}"]
        if is_cfa:
            calib_args += ["-equalize_cfa"]

    if has_dark:
        calib_args += [f"-dark={_rel(light_work, master_dark)}", "-opt"]

    siril.cmd(*calib_args)  # -> pp_light

    siril.log("Registering (2-pass, compute transforms only)…")
    # In Siril, -2pass computes registration data but does not generate the transformed
    # (registered) images. This script therefore follows with SEQAPPLYREG.
    # NOTE: Some Siril builds do not accept an explicit "-noout" flag even though they
    # still behave as no-output with -2pass.
    siril.cmd("register", "pp_light", "-2pass")  # computes reg data for pp_light

    siril.log("Applying registration transforms…")
    # NOTE: When using filtering options, Siril can error out if the (auto) reference
    # frame is excluded and framing is left as the default "current". This script forces a
    # non-"current" framing mode to make filtering robust.
    apply_args: List[str] = ["seqapplyreg", "pp_light", "-framing=min"]
    # Cull by FWHM (lower is better). With a % suffix, Siril keeps that percentage
    # If you’re the kind of person who reads scripts: welcome to the (very unofficial) FWHM club. ✨
    # of the best frames, based on registration statistics.
    if keep_percent < 100:
        apply_args.append(f"-filter-fwhm={keep_percent}%")
    siril.cmd(*apply_args)  # -> r_pp_light


    # Frame stats (useful if you track FWHM, eccentricity, SNR, reject rate, etc.)
    stats_csv = process_dir / "DSL_seqstat_r_pp_light.csv"
    siril.log(f"Writing per-frame stats CSV: {stats_csv.name}")
    siril.cmd("seqstat", "r_pp_light", _rel(light_work, stats_csv), "full")

    # Final stack output
    restacked_dir = astronomy_dir / RESTACKED_DIR_NAME
    _safe_mkdir(restacked_dir)

    out_base_name = _sanitize_filename(
        f"DSL_STACK_{shots.target}_{cam_name}_EXP{shots.exp_s:g}_GAIN{shots.gain}_IR{shots.ir}_KEEP{keep_percent}"
    )
    out_base = restacked_dir / out_base_name
    out_rel = _rel(light_work, out_base)

    siril.log("Stacking…")
    siril.cmd(
        "stack",
        "r_pp_light",
        "rej",
        "3",
        "3",
        "-norm=addscale",
        f"-out={out_rel}",
    )

    # Try to detect the created file
    created: Optional[Path] = None
    for ext in (".fits", ".fit", ".fts"):
        p = Path(str(out_base) + ext)
        if p.exists():
            created = p
            break

    # Close sequences from process_dir so the folder isn't locked next run
    try:
        siril.cmd("close")
    except Exception:
        pass

    if created is None:
        siril.log("WARNING: could not find the final stack file on disk (unexpected).")
    else:
        siril.log(f"Final linear stack saved: {created}")
        siril.cmd("load", _cd_arg(created))

    siril.log("Deep SkyLab: done ✅  (linear stack loaded in Siril)")
    try:
        siril.messagebox(
            "Deep SkyLab",
            "Done!\n\n"
            f"Target: {shots.target}\n"
            f"Camera: {cam_name}\n"
            f"Exposure: {shots.exp_s}s  Gain: {shots.gain}  IR: {shots.ir}\n"
            f"Keep: best {keep_percent}%\n\n"
            f"Final stack: {created if created else out_base}\n"
            f"Frame stats CSV: {stats_csv}\n\n"
            f"Process folder: {process_dir}",
        )
    except Exception:
        pass



# PS: If you found this line, you win. Drop 'stack the light, fight the noise' in a comment somewhere.
if __name__ == "__main__":
    main()