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
import time
import traceback

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

def load_fits_fz_single(path):
    """Lecture unitaire ultra-légère sans duplication mémoire."""
    with fits.open(path, mode="readonly", memmap=True) as hdul:
        hdu = None
        for current_hdu in hdul:
            if current_hdu.data is not None and isinstance(current_hdu.data, np.ndarray) and current_hdu.data.ndim >= 2:
                hdu = current_hdu
                break
        
        if hdu is None:
            raise ValueError(f"Aucune matrice d'image valide trouvée dans le fichier FITS : {path}")
            
        img = hdu.data.astype(np.float32)
        header = hdu.header
        
        C, H_in, W_in = img.shape
        crpix1 = float(header.get('CRPIX1', W_in / 2.0)) - 1.0
        crpix2 = float(header.get('CRPIX2', H_in / 2.0)) - 1.0
        
        return {
            'img': img,
            'dx': float(header.get('DX', 0.0)),
            'dy': float(header.get('DY', 0.0)),
            'angle_deg': float(header.get('ANGLE', 0.0)),
            'crpix1': crpix1,
            'crpix2': crpix2,
            'name': Path(path).name
        }

def gpu_bayer_drizzle_stack_sequential(fits_paths, scale=1.0, pixfrac=0.7, rgb_equal=True):
    device = get_torch_device()
    print(f"Périphérique : {device} (Normalisation par Canaux Indépendants - Mode Multi-Weight)")

    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # Chargement de la brute de référence
    ref_data = load_fits_fz_single(fits_paths[0])
    C, H_ref, W_ref = ref_data['img'].shape
    crpix1_ref, crpix2_ref = ref_data['crpix1'], ref_data['crpix2']

    H_out, W_out = int(H_ref * scale), int(W_ref * scale)
    crpix1_out, crpix2_out = crpix1_ref * scale, crpix2_ref * scale

    # Allocations mémoires Float32 requises par MPS Mac
    output_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    # Tableaux de compensation d'erreur de Kahan pour les longues séries (5000 images)
    output_compensation = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_compensation = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    r = (pixfrac * scale) / 2.0
    max_search = int(np.ceil(r)) + 1

    y_in, x_in = torch.meshgrid(torch.arange(H_ref, dtype=torch.float32, device=device), torch.arange(W_ref, dtype=torch.float32, device=device), indexing='ij')
    x_in_flat, y_in_flat = x_in.reshape(-1), y_in.reshape(-1)

    with torch.inference_mode():
        for i, path in enumerate(fits_paths):
            data = load_fits_fz_single(path)
            
            img = data['img']
            dx, dy, angle_deg = data['dx'], data['dy'], data['angle_deg']
            angle_rad = np.radians(angle_deg)
            crpix1_curr, crpix2_curr = data['crpix1'], data['crpix2']
            
            if (i + 1) % 100 == 0 or (i + 1) == len(fits_paths):
                print(f"[{i+1}/{len(fits_paths)}] Traitement -> {data['name']}")

            img_tensor = torch.as_tensor(img, device=device, dtype=torch.float32)

            # --- CALCUL DES DIMENSIONS REPLACÉ CORRECTEMENT ICI ---
            C_curr, H_in, W_in = img.shape

            if H_in != H_ref or W_in != W_ref:
                y_dyn, x_dyn = torch.meshgrid(torch.arange(H_in, dtype=torch.float32, device=device), torch.arange(W_in, dtype=torch.float32, device=device), indexing='ij')
                xf, yf = x_dyn.reshape(-1), y_dyn.reshape(-1)
                xc_in = xf - crpix1_curr
                yc_in = yf - crpix2_curr
            else:
                xc_in = x_in_flat - crpix1_curr
                yc_in = y_in_flat - crpix2_curr

            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            xc_scaled = (xc_in * cos_a - yc_in * sin_a) * scale
            yc_scaled = (xc_in * sin_a + yc_in * cos_a) * scale

            target_x = xc_scaled + dx + crpix1_out
            target_y = yc_scaled - dy + crpix2_out
            x1, x2 = target_x - r, target_x + r
            y1, y2 = target_y - r, target_y + r

            for dy_pix in range(-max_search, max_search + 1):
                out_y = torch.round(target_y) + dy_pix
                valid_y = (out_y >= 0) & (out_y < H_out)
                if not valid_y.any(): continue
                
                y_overlap = torch.clamp(torch.minimum(out_y, y2) - torch.maximum(out_y - 1.0, y1), min=0.0)
                
                for dx_pix in range(-max_search, max_search + 1):
                    out_x = torch.round(target_x) + dx_pix
                    valid_mask = valid_y & (out_x >= 0) & (out_x < W_out)
                    if not valid_mask.any(): continue

                    x_overlap = torch.clamp(torch.minimum(out_x, x2) - torch.maximum(out_x - 1.0, x1), min=0.0)
                    weights = x_overlap * y_overlap * valid_mask.float()

                    active_indices = weights > 0
                    if not active_indices.any(): continue

                    y_idx = out_y[active_indices].to(torch.int32)
                    x_idx = out_x[active_indices].to(torch.int32)
                    spatial_idx = y_idx * W_out + x_idx
                    w_act = weights[active_indices]

                    for channel in range(C):
                        flat_channel_input = img_tensor[channel].reshape(-1)
                        val_act = flat_channel_input[active_indices]

                        # 1. Somme de Kahan sur la carte de poids spécifique au canal
                        weight_input = torch.zeros(H_out * W_out, dtype=torch.float32, device=device)
                        weight_input.scatter_add_(0, spatial_idx, w_act)
                        
                        y_w = weight_input.view(H_out, W_out) - weight_compensation[channel]
                        t_w = weight_accum[channel] + y_w
                        weight_compensation[channel] = (t_w - weight_accum[channel]) - y_w
                        weight_accum[channel] = t_w

                        # 2. Somme de Kahan sur le signal du canal
                        val_input = torch.zeros(H_out * W_out, dtype=torch.float32, device=device)
                        val_input.scatter_add_(0, spatial_idx, val_act * w_act)

                        y_v = val_input.view(H_out, W_out) - output_compensation[channel]
                        t_v = output_accum[channel] + y_v
                        output_compensation[channel] = (t_v - output_accum[channel]) - y_v
                        output_accum[channel] = t_v

            del img_tensor

    print("Normalisation finale...")
    final_stack = torch.where(weight_accum > 0, output_accum / weight_accum, 0.0)

    if rgb_equal and C == 3:
        print("⚖️ Égalisation RVB...")
        masque_intersection_couleur = (weight_accum[0] > 0) & (weight_accum[1] > 0) & (weight_accum[2] > 0)
        seuil_poids_central = torch.max(weight_accum) * 0.8
        masque_centre_2d = weight_accum[0] > seuil_poids_central
        
        if not masque_centre_2d.any():
            masque_centre_2d = masque_intersection_couleur

        means = []
        for c in range(3):
            canal = final_stack[c]
            means.append(canal[masque_centre_2d].mean().item() if masque_centre_2d.any() else 1.0)
        
        k_red = means[1] / max(means[0], 1e-5)
        k_blue = means[1] / max(means[2], 1e-5)
        
        print(f"   -> Alignement couleur appliqué : R * {k_red:.4f} | B * {k_blue:.4f}")
        final_stack[0] *= k_red
        final_stack[2] *= k_blue

        for c in range(3):
            final_stack[c] = torch.where(masque_intersection_couleur, final_stack[c], torch.tensor(0.0, device=device))

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
        print("❌ Aucun fichier trouvé.")
    else:
        try:
            import time
            start_time = time.time()
            
            image_couleur, carte_poids = gpu_bayer_drizzle_stack_sequential(fichiers_trouves, scale=1.0, rgb_equal=True)
            
            fits.writeto("drizzle_final_perfect.fits", image_couleur, overwrite=True)
            print(f"🎉 [MULTI-WEIGHT KAHAN STACK] Traitement achevé en {time.time() - start_time:.2f} secondes !")
        except Exception as e:
            print("\n💥 Une erreur globale est survenue durant l'exécution !")
            traceback.print_exc()