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

def gpu_bayer_drizzle_stack_one_pass(fits_paths, scale=1.0, pixfrac=0.7, sigma_high=3.0, sigma_low=3.0):
    """
    Bayer Drizzle Stacking en UNE SEULE PASSE sur GPU.
    Utilise une variante de l'algorithme glissant de Welford pour le calcul dynamique du Sigma
    et rejette les satellites en temps réel à la volée.
    """
    device = get_torch_device()
    print(f"Périphérique de calcul : {device}")
    print(f"Mode 1 Passe -> Scale: {scale}x | Pixfrac: {pixfrac} | Rejet Sigma: {sigma_high}")

    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # 1. Lecture de l'image de référence pour les dimensions globales
    with fits.open(fits_paths[0]) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        C, H_ref, W_ref = hdu.data.shape
        crpix1_ref = float(hdu.header.get('CRPIX1', W_ref / 2.0)) - 1.0
        crpix2_ref = float(hdu.header.get('CRPIX2', H_ref / 2.0)) - 1.0

    H_out, W_out = int(H_ref * scale), int(W_ref * scale)
    crpix1_out = crpix1_ref * scale
    crpix2_out = crpix2_ref * scale

    # Accumulateurs principaux pour le résultat final
    output_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    # Accumulateurs glissants pour l'évaluation statistique en temps réel (Algorithme de Welford)
    # n_images tracks the current valid weight/count per pixel location
    M2_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    r = (pixfrac * scale) / 2.0
    max_search = int(np.ceil(r)) + 1

    # 2. Boucle unique sur les fichiers FITS
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

            print(f"[{i+1}/{len(fits_paths)}] Passe Unique (Drizzle + Rejet Flottant) -> {Path(path).name}")
            img_tensor = torch.tensor(img, dtype=torch.float32, device=device)

        # Génération géométrique
        y_in, x_in = torch.meshgrid(torch.arange(H_in, device=device), torch.arange(W_in, device=device), indexing='ij')
        x_in_flat, y_in_flat = x_in.reshape(-1), y_in.reshape(-1)
        xc_out = ( (x_in_flat - crpix1_curr) * np.cos(angle_rad) - (y_in_flat - crpix2_curr) * np.sin(angle_rad) + dx ) * scale
        yc_out = ( (x_in_flat - crpix1_curr) * np.sin(angle_rad) + (y_in_flat - crpix2_curr) * np.cos(angle_rad) + dy ) * scale
        target_x, target_y = xc_out + crpix1_out, yc_out + crpix2_out
        x1, x2, y1, y2 = target_x - r, target_x + r, target_y - r, target_y + r

        for dy_pix in range(-max_search, max_search + 1):
            for dx_pix in range(-max_search, max_search + 1):
                out_x, out_y = torch.round(target_x) + dx_pix, torch.round(target_y) + dy_pix
                valid_mask = (out_x >= 0) & (out_x < W_out) & (out_y >= 0) & (out_y < H_out)
                if not valid_mask.any(): continue

                y_overlap = torch.clamp(torch.minimum(out_y, y2) - torch.maximum(out_y - 1.0, y1), min=0.0)
                x_overlap = torch.clamp(torch.minimum(out_x, x2) - torch.maximum(out_x - 1.0, x1), min=0.0)
                weights = x_overlap * y_overlap * valid_mask.float()
                active_indices = weights > 0
                if not active_indices.any(): continue

                y_idx = out_y[active_indices].long()
                x_idx = out_x[active_indices].long()
                flat_spatial_indices = y_idx * W_out + x_idx

                for channel in range(C_curr):
                    flat_channel_input = img_tensor[channel].reshape(-1)
                    actual_vals = flat_channel_input[active_indices]
                    w_active = weights[active_indices]

                    # Extraction de l'état statistique actuel de la pile pour ces coordonnées
                    current_weights = weight_accum[channel].view(-1)[flat_spatial_indices]
                    current_means = output_accum[channel].view(-1)[flat_spatial_indices]
                    current_M2 = M2_accum[channel].view(-1)[flat_spatial_indices]

                    # Calcul de l'écart-type glissant actuel (σ)
                    # Si on a moins de 2 images d'historique de poids, on met un sigma par défaut
                    current_sigmas = torch.sqrt(torch.clamp(current_M2 / torch.clamp(current_weights, min=1.0), min=1e-5))

                    # --- REJET STATISTIQUE EN PASSE UNIQUE ---
                    is_not_black = (actual_vals > 0.0)
                    
                    # On n'applique le rejet qu'à partir du moment où le pixel a déjà reçu au moins 2 images d'historique
                    # pour éviter de rejeter les premières poses arbitrairement
                    has_history = (current_weights >= 1.5) 
                    
                    within_bounds = ~has_history | (
                        (actual_vals >= (current_means - sigma_low * current_sigmas)) & \
                        (actual_vals <= (current_means + sigma_high * current_sigmas))
                    )

                    valid_pixel_mask = active_indices.clone()
                    valid_pixel_mask[active_indices] = is_not_black & within_bounds
                    
                    if not valid_pixel_mask.any(): continue

                    # Ré-extraction filtrée des pixels acceptés
                    w_act = weights[valid_pixel_mask]
                    val_act = flat_channel_input[valid_pixel_mask]
                    spatial_idx = out_y[valid_pixel_mask].long() * W_out + out_x[valid_pixel_mask].long()

                    # --- MISE À JOUR FORMELLES DE WELFORD ET ACCUMULATION ---
                    # Version vectorisée des équations glissantes
                    old_means = output_accum[channel].view(-1)[spatial_idx]
                    old_weights = weight_accum[channel].view(-1)[spatial_idx]
                    new_weights = old_weights + w_act

                    # Nouvelle moyenne pondérée glissante
                    # Moyenne = Moyenne_précédente + (Valeur - Moyenne_précédente) * (Poids_courant / Poids_total)
                    delta = val_act - old_means
                    new_means = old_means + delta * (w_act / torch.clamp(new_weights, min=1e-5))

                    # Mise à jour de la somme des carrés des différences (M2) pour le sigma de la prochaine frame
                    delta2 = val_act - new_means
                    welford_M2_update = w_act * delta * delta2

                    # Injection atomique sur la carte graphique
                    output_accum[channel].view(-1).scatter_add_(0, spatial_idx, val_act * w_act)
                    weight_accum[channel].view(-1).scatter_add_(0, spatial_idx, w_act)
                    M2_accum[channel].view(-1).scatter_add_(0, spatial_idx, welford_M2_update)

    # 3. Normalisation finale
    print("Normalisation finale de la matrice en passe unique...")
    final_stack = torch.where(weight_accum > 0, output_accum / weight_accum, 0.0)
    return final_stack.cpu().numpy(), weight_accum.cpu().numpy()

# --- BLOC MAIN ---
if __name__ == "__main__":
    from pathlib import Path
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
            image_couleur, carte_poids = gpu_bayer_drizzle_stack_one_pass(
                fichiers_trouves, 
                scale=1.0, 
                pixfrac=0.7,
                sigma_high=3.0,
                sigma_low=3.0
            )
            fits.writeto("drizzle_1pass_cleaned.fits", image_couleur, overwrite=True)
            print("🎉 Image finale sauvegardée avec succès sous : 'drizzle_1pass_cleaned.fits'")
        except Exception as e:
            print(f"💥 Erreur lors de l'exécution : {e}")