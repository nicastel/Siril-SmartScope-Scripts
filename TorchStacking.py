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

def gpu_bayer_drizzle_stack_torch(fits_paths, scale=1.0, pixfrac=0.7):
    """
    Empilement Drizzle PyTorch universel [C, H, W] avec facteur d'échelle optionnel.
    
    Parameters:
      fits_paths: Liste des chemins vers les fichiers FITS.
      scale: Facteur d'upscale. Utilisez 1.0 pour garder la résolution d'origine de Siril,
             ou 2.0 pour doubler la résolution de l'image finale.
      pixfrac: Taille de la goutte de pixel (footprint), généralement entre 0.6 et 0.8.
    """
    device = get_torch_device()
    print(f"Périphérique de calcul activé : {device}")
    print(f"Configuration -> Scale (Upscale): {scale}x | Pixfrac (Drop size): {pixfrac}")

    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # 1. Utiliser le premier fichier pour définir les dimensions de référence
    with fits.open(fits_paths[0]) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        C, H_ref, W_ref = hdu.data.shape
        
        # Récupération du pixel de référence (CRPIX) de l'image maîtresse
        crpix1_ref = float(hdu.header.get('CRPIX1', W_ref / 2.0)) - 1.0
        crpix2_ref = float(hdu.header.get('CRPIX2', H_ref / 2.0)) - 1.0

    # Calcul des dimensions globales de sortie indexées sur le paramètre 'scale'
    H_out, W_out = int(H_ref * scale), int(W_ref * scale)
    crpix1_out = crpix1_ref * scale
    crpix2_out = crpix2_ref * scale

    # Allocation des accumulateurs sur le GPU/CPU actif
    output_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    # Calcul de la taille de la "goutte" (footprint) projetée
    r = (pixfrac * scale) / 2.0

    # 2. Boucle sur la séquence de fichiers FITS
    for i, path in enumerate(fits_paths):
        with fits.open(path) as hdul:
            # Gestion automatique des fichiers compressés .fz où la donnée est dans l'extension 1
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            img = hdu.data
            header = hdu.header
            
            C_curr, H_in, W_in = img.shape
            
            # Paramètres de transformation Siril
            dx = float(header.get('DX', 0.0))
            dy = float(header.get('DY', 0.0))
            angle_deg = float(header.get('ANGLE', 0.0))
            angle_rad = np.radians(angle_deg)
            
            # Lecture du pixel de référence de l'image courante
            crpix1_curr = float(header.get('CRPIX1', W_in / 2.0)) - 1.0
            crpix2_curr = float(header.get('CRPIX2', H_in / 2.0)) - 1.0
            
            print(f"[{i+1}/{len(fits_paths)}] Traitement -> {Path(path).name} | DX: {dx:.2f}, DY: {dy:.2f}")

            img_tensor = torch.tensor(img, dtype=torch.float32, device=device)

        # Génération de la grille 2D courante
        y_in, x_in = torch.meshgrid(
            torch.arange(H_in, dtype=torch.float32, device=device),
            torch.arange(W_in, dtype=torch.float32, device=device),
            indexing='ij'
        )
        x_in_flat = x_in.reshape(-1)
        y_in_flat = y_in.reshape(-1)

        # Centrage par rapport au pixel de référence
        xc_in = x_in_flat - crpix1_curr
        yc_in = y_in_flat - crpix2_curr

        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)

        # Transformation géométrique Drizzle
        xc_out = (xc_in * cos_t - yc_in * sin_t + dx) * scale
        yc_out = (xc_in * sin_t + yc_in * cos_t + dy) * scale

        # Repositionnement dans le repère de sortie
        target_x = xc_out + crpix1_out
        target_y = yc_out + crpix2_out

        x1, x2 = target_x - r, target_x + r
        y1, y2 = target_y - r, target_y + r

        # Détermination de la zone locale d'impact
        max_search = int(np.ceil(r)) + 1
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

                # Filtrage canal par canal (Inclusion du filtre pixel noir)
                for channel in range(C_curr):
                    flat_channel_input = img_tensor[channel].reshape(-1)
                    
                    valid_pixel_mask = active_indices.clone()
                    valid_pixel_mask[active_indices] = (flat_channel_input[active_indices] > 0.0)
                    
                    if not valid_pixel_mask.any():
                        continue
                        
                    w_active = weights[valid_pixel_mask]
                    val_active = flat_channel_input[valid_pixel_mask] * w_active
                    
                    y_idx = out_y[valid_pixel_mask].long()
                    x_idx = out_x[valid_pixel_mask].long()
                    flat_spatial_indices = y_idx * W_out + x_idx

                    output_accum[channel].view(-1).scatter_add_(0, flat_spatial_indices, val_active)
                    weight_accum[channel].view(-1).scatter_add_(0, flat_spatial_indices, w_active)

    print("Normalisation de la carte géométrique finale...")
    final_stack = torch.where(weight_accum > 0, output_accum / weight_accum, 0.0)
    
    return final_stack.cpu().numpy(), weight_accum.cpu().numpy()


# --- SECTION MAIN ADAPTÉE AUX MULTI-EXTENSIONS DU RÉPERTOIRE COURANT ---
if __name__ == "__main__":
    extensions_valides = {".fit", ".fits", ".fz"}
    repertoire_courant = Path(".")
    fichiers_trouves = []

    for fichier in repertoire_courant.iterdir():
        if fichier.is_file() and fichier.name.lower().startswith("r_"):
            suffixes = [s.lower() for s in fichier.suffixes]
            if len(suffixes) >= 1 and suffixes[-1] in extensions_valides:
                if suffixes[-1] == ".fz":
                    if len(suffixes) >= 2 and suffixes[-2] in {".fit", ".fits"}:
                        fichiers_trouves.append(str(fichier))
                else:
                    fichiers_trouves.append(str(fichier))

    fichiers_trouves.sort()

    if not fichiers_trouves:
        print("❌ Aucun fichier correspondant trouvé.")
    else:
        print(f"🚀 {len(fichiers_trouves)} fichiers détectés pour le Drizzle.")
        
        # --- CONFIGURATION DE L'UPSCALE ICI ---
        # scale = 1.0 -> Même résolution que Siril d'origine (Recommandé si échantillonnage OK)
        # scale = 2.0 -> Résolution doublée (Utile en cas de sous-échantillonnage sévère)
        FACTEUR_UPSCALE = 1.0  
        TAILLE_GOUTTE = 0.7 if FACTEUR_UPSCALE == 1.0 else 0.6
        
        try:
            image_couleur, carte_poids = gpu_bayer_drizzle_stack_torch(
                fichiers_trouves, 
                scale=FACTEUR_UPSCALE, 
                pixfrac=TAILLE_GOUTTE
            )
            
            nom_sortie = "drizzle_final_output.fits"
            fits.writeto(nom_sortie, image_couleur, overwrite=True)
            print(f"🎉 Image finale sauvegardée avec succès sous : '{nom_sortie}'")
            
        except Exception as e:
            print(f"💥 Erreur lors de l'exécution : {e}")

