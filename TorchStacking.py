import sirilpy as s
import sys
s.ensure_installed("numpy", "astropy")

th = s.TorchHelper()
th.ensure_torch()

# 3. DirectML fallback per Windows senza CUDA/XPU
if sys.platform == "win32":
    try:
        import torch
        if not torch.cuda.is_available() and not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            s.ensure_installed("torch-directml")
    except Exception:
        pass


import torch
import numpy as np
from astropy.io import fits
from pathlib import Path

def get_torch_device() -> torch.device:
    global _dml_device
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    if sys.platform == 'win32':
        try:
            import torch_directml
            _dml_device = torch_directml.device()
            return _dml_device
        except Exception:
            pass
    return torch.device('cpu')

# --- KERNEL FUSIONNÉ ET COMPILÉ ---
@torch.compile(dynamic=False, fullgraph=False)
def process_pixel_footprint_gpu(img_tensor, output_accum, weight_accum, M2_accum, 
                                target_x, target_y, r, C_curr, H_out, W_out, 
                                sigma_low, sigma_high, max_search):
    x1, x2 = target_x - r, target_x + r
    y1, y2 = target_y - r, target_y + r

    for dy_pix in range(-max_search, max_search + 1):
        for dx_pix in range(-max_search, max_search + 1):
            out_x = torch.round(target_x) + dx_pix
            out_y = torch.round(target_y) + dy_pix

            valid_mask = (out_x >= 0) & (out_x < W_out) & (out_y >= 0) & (out_y < H_out)
            if not valid_mask.any(): 
                continue

            y_overlap = torch.clamp(torch.minimum(out_y, y2) - torch.maximum(out_y - 1.0, y1), min=0.0)
            x_overlap = torch.clamp(torch.minimum(out_x, x2) - torch.maximum(out_x - 1.0, x1), min=0.0)
            weights = x_overlap * y_overlap * valid_mask.float()
            active_indices = weights > 0
            if not active_indices.any(): 
                continue

            y_idx = out_y[active_indices].long()
            x_idx = out_x[active_indices].long()
            flat_spatial_indices = y_idx * W_out + x_idx

            for channel in range(C_curr):
                flat_channel_input = img_tensor[channel].reshape(-1)
                actual_vals = flat_channel_input[active_indices]
                w_active = weights[active_indices]

                current_weights = weight_accum[channel].view(-1)[flat_spatial_indices]
                current_means = output_accum[channel].view(-1)[flat_spatial_indices]
                current_M2 = M2_accum[channel].view(-1)[flat_spatial_indices]

                current_sigmas = torch.sqrt(torch.clamp(current_M2 / torch.clamp(current_weights, min=1.0), min=1e-5))

                is_not_black = (actual_vals > 0.0)
                has_history = (current_weights >= 1.5) 
                
                within_bounds = ~has_history | (
                    (actual_vals >= (current_means - sigma_low * current_sigmas)) & \
                    (actual_vals <= (current_means + sigma_high * current_sigmas))
                )

                valid_pixel_mask = active_indices.clone()
                valid_pixel_mask[active_indices] = is_not_black & within_bounds
                if not valid_pixel_mask.any(): 
                    continue

                w_act = weights[valid_pixel_mask]
                val_act = flat_channel_input[valid_pixel_mask]
                spatial_idx = out_y[valid_pixel_mask].long() * W_out + out_x[valid_pixel_mask].long()

                old_means = output_accum[channel].view(-1)[spatial_idx]
                old_weights = weight_accum[channel].view(-1)[spatial_idx]
                new_weights = old_weights + w_act

                delta = val_act - old_means
                new_means = old_means + delta * (w_act / torch.clamp(new_weights, min=1e-5))
                delta2 = val_act - new_means
                welford_M2_update = w_act * delta * delta2

                output_accum[channel].view(-1).scatter_add_(0, spatial_idx, val_act * w_act)
                weight_accum[channel].view(-1).scatter_add_(0, spatial_idx, w_act)
                M2_accum[channel].view(-1).scatter_add_(0, spatial_idx, welford_M2_update)


def gpu_bayer_drizzle_stack_compiled(fits_paths, scale=1.0, pixfrac=0.7, sigma_high=3.0, sigma_low=3.0, rgb_equal=True):
    device = get_torch_device()
    print(f"Périphérique : {device} (Mode Compilé Haute Vitesse)")

    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # --- CORRECTION ICI : Lecture du premier chemin indexé [0] ---
    with fits.open(fits_paths[0]) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        C, H_ref, W_ref = hdu.data.shape
        crpix1_ref = float(hdu.header.get('CRPIX1', W_ref / 2.0)) - 1.0
        crpix2_ref = float(hdu.header.get('CRPIX2', H_ref / 2.0)) - 1.0

    H_out, W_out = int(H_ref * scale), int(W_ref * scale)
    crpix1_out = crpix1_ref * scale
    crpix2_out = crpix2_ref * scale

    output_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    M2_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    r = (pixfrac * scale) / 2.0
    max_search = int(np.ceil(r)) + 1

    # Initialisation de la grille de coordonnées d'origine
    y_in, x_in = torch.meshgrid(torch.arange(H_ref, device=device), torch.arange(W_ref, device=device), indexing='ij')
    x_in_flat, y_in_flat = x_in.reshape(-1), y_in.reshape(-1)

    for i, path in enumerate(fits_paths):
        with fits.open(path) as hdul:
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            img = hdu.data
            header = hdu.header
            C_curr, H_in, W_in = img.shape
            
            dx = float(header.get('DX', 0.0))
            dy = float(header.get('DY', 0.0))
            angle_rad = np.radians(float(header.get('ANGLE', 0.0)))
            crpix1_curr = float(header.get('CRPIX1', W_in / 2.0)) - 1.0
            crpix2_curr = float(header.get('CRPIX2', H_in / 2.0)) - 1.0

            img_tensor = torch.tensor(img, dtype=torch.float32, device=device)

        # Regénération dynamique si Siril a légèrement recadré cette image précise
        if H_in != H_ref or W_in != W_ref:
            y_in_dyn, x_in_dyn = torch.meshgrid(torch.arange(H_in, device=device), torch.arange(W_in, device=device), indexing='ij')
            x_flat, y_flat = x_in_dyn.reshape(-1), y_in_dyn.reshape(-1)
        else:
            x_flat, y_flat = x_in_flat, y_in_flat

        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        xc_out = ( (x_flat - crpix1_curr) * cos_a - (y_flat - crpix2_curr) * sin_a + dx ) * scale
        yc_out = ( (x_flat - crpix1_curr) * sin_a + (y_flat - crpix2_curr) * cos_a + dy ) * scale
        target_x, target_y = xc_out + crpix1_out, yc_out + crpix2_out

        process_pixel_footprint_gpu(
            img_tensor, output_accum, weight_accum, M2_accum,
            target_x, target_y, r, C_curr, H_out, W_out,
            sigma_low, sigma_high, max_search
        )
        print(f"[{i+1}/{len(fits_paths)}] Injecté dans le pipeline matériel GPU.")

    print("Normalisation globale...")
    final_stack = torch.where(weight_accum > 0, output_accum / weight_accum, 0.0)

    if rgb_equal and C == 3:
        print("⚖️ Égalisation RVB finale...")
        mean_green = final_stack[1][final_stack[1] > 0.0].mean()
        mean_red = final_stack[0][final_stack[0] > 0.0].mean()
        mean_blue = final_stack[2][final_stack[2] > 0.0].mean()

        final_stack[0] *= (mean_green / torch.clamp(mean_red, min=1e-5))
        final_stack[2] *= (mean_green / torch.clamp(mean_blue, min=1e-5))

    return final_stack.cpu().numpy(), weight_accum.cpu().numpy()


if __name__ == "__main__":
    extensions_valides = {".fit", ".fits", ".fz"}
    repertoire_courant = Path(".")
    fichiers_trouves = [
        str(f) for f in repertoire_courant.iterdir()
        if f.is_file() and f.name.lower().startswith("r_") and 
        (f.suffixes[-1].lower() in extensions_valides or (len(f.suffixes) >= 2 and f.suffixes[-1].lower() == ".fz" and f.suffixes[-2].lower() in {".fit", ".fits"}))
    ]
    fichiers_trouves.sort()

    if not fichiers_trouves:
        print("❌ Aucun fichier correspondant trouvé.")
    else:
        try:
            image_couleur, carte_poids = gpu_bayer_drizzle_stack_compiled(
                fichiers_trouves, scale=1.0, pixfrac=0.7, rgb_equal=True
            )
            fits.writeto("drizzle_speed_optimized.fits", image_couleur, overwrite=True)
            print("🎉 Image sauvegardée sous : 'drizzle_speed_optimized.fits'")
        except Exception as e:
            print(f"💥 Erreur : {e}")