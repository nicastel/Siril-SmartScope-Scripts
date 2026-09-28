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
    """Lecture unitaire brute ultra-rapide (ZÉRO normalisation, parfait pour le SPCC)."""
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

def gpu_linear_dynamic_boundary_stack(fits_paths, scale=1.0, pixfrac=1.0):
    device = get_torch_device()
    print(f"Périphérique : {device} (Mode Linéaire Brut 1-Passe | Alignement Géométrique Siril)")

    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # 1. Initialisation sur la première image de référence
    ref_data = load_fits_fz_single(fits_paths[0])
    C, H_ref, W_ref = ref_data['img'].shape
    
    H_out, W_out = int(H_ref * scale), int(W_ref * scale)
    
    x_offset_canvas = 0
    y_offset_canvas = 0
    crpix1_out = ref_data['crpix1'] * scale
    crpix2_out = ref_data['crpix2'] * scale

    output_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    r = (pixfrac * scale) / 2.0
    max_search = 1 

    with torch.inference_mode():
        for i, path in enumerate(fits_paths):
            data = load_fits_fz_single(path)
            img = data['img']
            dx, dy, angle_deg = data['dx'], data['dy'], data['angle_deg']
            
            # --- CORRECTION GÉOMÉTRIQUE APPLIQUÉE ICI ---
            # L'inversion du signe (-) compense le repère Top-Down de PyTorch
            # et recalcule la rotation exacte dans le sens de Siril.
            angle_rad = -np.radians(angle_deg)
            crpix1_curr, crpix2_curr = data['crpix1'], data['crpix2']
            
            if (i + 1) % 100 == 0 or (i + 1) == len(fits_paths):
                print(f"[{i+1}/{len(fits_paths)}] Empilement linéaire -> {data['name']}")

            img_tensor = torch.as_tensor(img, device=device, dtype=torch.float32)
            _, H_in, W_in = img.shape

            y_dyn, x_dyn = torch.meshgrid(torch.arange(H_in, dtype=torch.float32, device=device), torch.arange(W_in, dtype=torch.float32, device=device), indexing='ij')
            xc_scaled = (x_dyn.reshape(-1) - crpix1_curr) * scale
            yc_scaled = (y_dyn.reshape(-1) - crpix2_curr) * scale

            # Application de la rotation corrigée
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            xc_rot = xc_scaled * cos_a - yc_scaled * sin_a
            yc_rot = xc_scaled * sin_a + yc_scaled * cos_a

            # Alignement des translations avec l'offset dynamique du canvas large
            target_x = xc_rot + ((dx + x_offset_canvas) * scale) + crpix1_out
            target_y = yc_rot - ((dy - y_offset_canvas) * scale) + crpix2_out

            # --- EXPANSION DE LA TOILE À LA VOLÉE ---
            min_x_need = int(torch.floor(target_x.min()).item()) - max_search
            max_x_need = int(torch.ceil(target_x.max()).item()) + max_search
            min_y_need = int(torch.floor(target_y.min()).item()) - max_search
            max_y_need = int(torch.ceil(target_y.max()).item()) + max_search

            pad_left = max(0, -min_x_need)
            pad_right = max(0, max_x_need - W_out + 1)
            pad_top = max(0, -min_y_need)
            pad_bottom = max(0, max_y_need - H_out + 1)

            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                x_offset_canvas += pad_left
                y_offset_canvas += pad_top
                crpix1_out += pad_left
                crpix2_out += pad_top
                
                output_accum = F.pad(output_accum, (pad_left, pad_right, pad_top, pad_bottom))
                weight_accum = F.pad(weight_accum, (pad_left, pad_right, pad_top, pad_bottom))
                
                H_out, W_out = output_accum.shape[1:]
                target_x += pad_left
                target_y += pad_top

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

                    spatial_idx = out_y[active_indices].to(torch.int32) * W_out + out_x[active_indices].to(torch.int32)
                    w_act = weights[active_indices]

                    for channel in range(C):
                        flat_channel_input = img_tensor[channel].reshape(-1)
                        val_act = flat_channel_input[active_indices]

                        output_accum[channel].view(-1).scatter_add_(0, spatial_idx, val_act * w_act)
                        weight_accum[channel].view(-1).scatter_add_(0, spatial_idx, w_act)

            del img_tensor

    print("Normalisation finale...")
    final_stack = torch.where(weight_accum > 0, output_accum / weight_accum, 0.0)

    # --- COMPORTEMENT SIRIL BRUT : Pas de masque d'intersection destructeur ---
    # L'image conserve l'ensemble des données empilées brutes prêtes pour le SPCC
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
        print("❌ Aucun fichier trouvé dans le répertoire.")
    else:
        try:
            import time
            start_time = time.time()
            image_couleur, carte_poids = gpu_linear_dynamic_boundary_stack(fichiers_trouves, scale=1.0, pixfrac=1.0)
            fits.writeto("drizzle_final_maximum_boundary.fits", image_couleur, overwrite=True)
            print(f"🎉 [LINEAL STACK SIRIL COMPATIBLE COMPLETE] Traitement achevé en {time.time() - start_time:.2f} secondes !")
        except Exception as e:
            print("\n💥 Une erreur globale est survenue durant l'exécution !")
            traceback.print_exc()
