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
import torch.nn.functional as F
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

def gpu_bayer_drizzle_stack_fixed(fits_paths, scale=1.0, pixfrac=0.7, sigma_high=3.0, sigma_low=3.0, rgb_equal=True):
    """
    Pipeline Drizzle unique : Précision géométrique absolue Siril + Optimisation I/O.
    Garantit des étoiles parfaitement rondes sans artefacts.
    """
    device = get_torch_device()
    print(f"Périphérique : {device} (Mode Géométrique Certifié)")

    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # 1. Lecture stricte du premier fichier FITS pour initialiser les géométries
    with fits.open(fits_paths[0], mode="readonly", memmap=True) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        C, H_ref, W_ref = hdu.data.shape
        crpix1_ref = float(hdu.header.get('CRPIX1', W_ref / 2.0)) - 1.0
        crpix2_ref = float(hdu.header.get('CRPIX2', H_ref / 2.0)) - 1.0

    H_out, W_out = int(H_ref * scale), int(W_ref * scale)
    crpix1_out = crpix1_ref * scale
    crpix2_out = crpix2_ref * scale

    # Allocations VRAM
    output_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    M2_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    r = (pixfrac * scale) / 2.0
    max_search = int(np.ceil(r)) + 1

    # Activation du mode inférence de PyTorch pour couper l'overhead CPU
    with torch.inference_mode():
        for i, path in enumerate(fits_paths):
            with fits.open(path, mode="readonly", memmap=True) as hdul:
                hdu = hdul[0] if hdul[0].data is not None else hdul[1]
                img = hdu.data
                header = hdu.header
                
                C_curr, H_in, W_in = img.shape
                dx = float(header.get('DX', 0.0))
                dy = float(header.get('DY', 0.0))
                angle_rad = np.radians(float(header.get('ANGLE', 0.0)))
                crpix1_curr = float(header.get('CRPIX1', W_in / 2.0)) - 1.0
                crpix2_curr = float(header.get('CRPIX2', H_in / 2.0)) - 1.0

                # Lecture par bloc mmap directe vers la VRAM
                img_tensor = torch.as_tensor(img, dtype=torch.float32, device=device)

            # Génération de la grille spatiale locale à l'image courante
            y_in, x_in = torch.meshgrid(
                torch.arange(H_in, dtype=torch.float32, device=device),
                torch.arange(W_in, dtype=torch.float32, device=device),
                indexing='ij'
            )
            x_in_flat = x_in.reshape(-1)
            y_in_flat = y_in.reshape(-1)

            # Centrage absolu sur le point pivot intrinsèque calculé par Siril
            xc_in = x_in_flat - crpix1_curr
            yc_in = y_in_flat - crpix2_curr

            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Application rigoureuse de la matrice de rotation et de translation
            xc_out = (xc_in * cos_a - yc_in * sin_a + dx) * scale
            yc_out = (xc_in * sin_a + yc_in * cos_a + dy) * scale

            target_x = xc_out + crpix1_out
            target_y = yc_out + crpix2_out

            x1, x2 = target_x - r, target_x + r
            y1, y2 = target_y - r, target_y + r

            # Projection discrète (Drizzle physique) sur la matrice de pixels de sortie
            for dy_pix in range(-max_search, max_search + 1):
                for dx_pix in range(-max_search, max_search + 1):
                    out_x = torch.round(target_x) + dx_pix
                    out_y = torch.round(target_y) + dy_pix

                    valid_mask = (out_x >= 0) & (out_x < W_out) & (out_y >= 0) & (out_y < H_out)
                    if not valid_mask.any(): 
                        continue

                    # Calcul précis de la surface d'intersection (Overlap)
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

            if (i + 1) % 20 == 0 or (i + 1) == len(fits_paths):
                print(f"✔️ [{i+1}/{len(fits_paths)}] Images traitées.")

    print("Normalisation finale de la pile...")
    final_stack = torch.where(weight_accum > 0, output_accum / weight_accum, 0.0)

    # --- ÉGALISATION DES HISTOGRAMMES RVB ---
    if rgb_equal and C == 3:
        print("⚖️ Égalisation RVB (Balance des Blancs matérielle)...")
        means = []
        for c in range(3):
            canal = final_stack[c]
            mask = canal > 0.0
            means.append(canal[mask].mean().item() if mask.any() else 1.0)
        
        mean_red, mean_green, mean_blue = means[0], means[1], means[2]

        k_red = mean_green / max(mean_red, 1e-5)
        k_blue = mean_green / max(mean_blue, 1e-5)

        print(f"   -> Application des coefficients : R * {k_red:.4f} | B * {k_blue:.4f}")
        final_stack[0] *= k_red
        final_stack[2] *= k_blue

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
            import time
            start_time = time.time()
            
            # Paramètres de référence : scale=1.0 (optionnel 2.0), pixfrac=0.7
            image_couleur, carte_poids = gpu_bayer_drizzle_stack_fixed(
                fichiers_trouves, scale=1.0, pixfrac=0.7, rgb_equal=True
            )
            
            fits.writeto("drizzle_fixed_final.fits", image_couleur, overwrite=True)
            print(f"🎉 Traitement achevé avec succès en {time.time() - start_time:.2f} secondes !")
        except Exception as e:
            print(f"💥 Erreur globale : {e}")