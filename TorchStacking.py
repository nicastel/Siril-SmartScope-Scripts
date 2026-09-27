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
import glob

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

def gpu_bayer_drizzle_stack_torch(fits_paths, scale=2.0, pixfrac=0.6):
    """
    Empilement Drizzle PyTorch universel [C, H, W] corrigé.
    Gère dynamiquement les variations de dimensions (recadrage Siril).
    """
    device = get_torch_device()
    print(f"Périphérique de calcul activé : {device}")

    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # 1. Utiliser le premier fichier pour définir les dimensions de SORTIE de référence
    with fits.open(fits_paths[0]) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        C, H_ref, W_ref = hdu.data.shape

    H_out, W_out = int(H_ref * scale), int(W_ref * scale)

    # Accumulateurs 3D [C, H_out, W_out] sur le GPU actif
    output_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    r = (pixfrac * scale) / 2.0

    # 2. Boucle sur les fichiers FITS
    for i, path in enumerate(fits_paths):
        with fits.open(path) as hdul:
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            img = hdu.data  # Forme [C, H, W] locale à cette image
            header = hdu.header
            
            # Dimensions réelles de l'image courante
            C_curr, H_in, W_in = img.shape
            
            dx = float(header.get('DX', 0.0))
            dy = float(header.get('DY', 0.0))
            angle_deg = float(header.get('ANGLE', 0.0))
            angle_rad = np.radians(angle_deg)
            
            print(f"[{i+1}/{len(fits_paths)}] PyTorch Stack -> {path} | Taille: [{C_curr}x{H_in}x{W_in}] | DX: {dx:.2f}, DY: {dy:.2f}")

            # Envoi des données 3D sur le GPU et aplatissement complet en 1D
            img_tensor = torch.tensor(img, dtype=torch.float32, device=device).reshape(-1)

        # --- GÉNÉRATION DYNAMIQUE DES GRILLES POUR L'IMAGE COURANTE ---
        y_in, x_in = torch.meshgrid(
            torch.arange(H_in, dtype=torch.float32, device=device),
            torch.arange(W_in, dtype=torch.float32, device=device),
            indexing='ij'
        )

        x_in_flat = x_in.reshape(-1)
        y_in_flat = y_in.reshape(-1)

        # Duplication des coordonnées pour les répéter sur les C canaux réels de cette image
        num_pixels = H_in * W_in
        x_in_flat_3d = x_in_flat.repeat(C_curr)
        y_in_flat_3d = y_in_flat.repeat(C_curr)
        
        # Tenseur des canaux associés : [0,0,..0, 1,1,..1, 2,2,..2]
        channels_flat = torch.arange(C_curr, device=device).repeat_interleave(num_pixels)

        # Coordonnées centrées basées sur la taille intrinsèque de l'image courante
        xc_in = x_in_flat_3d - W_in / 2.0
        yc_in = y_in_flat_3d - H_in / 2.0

        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)

        # Transformation géométrique globale projetée vers le repère de SORTIE fixe
        xc_out = (xc_in * cos_t - yc_in * sin_t + dx) * scale
        yc_out = (xc_in * sin_t + yc_in * cos_t + dy) * scale

        target_x = xc_out + W_out / 2.0
        target_y = yc_out + H_out / 2.0

        x1, x2 = target_x - r, target_x + r
        y1, y2 = target_y - r, target_y + r

        max_search = int(np.ceil(r)) + 1
        for dy_pix in range(-max_search, max_search + 1):
            for dx_pix in range(-max_search, max_search + 1):
                
                out_x = torch.round(target_x) + dx_pix
                out_y = torch.round(target_y) + dy_pix

                # Le masque valide utilise désormais les dimensions globales fixes H_out et W_out
                valid_mask = (out_x >= 0) & (out_x < W_out) & (out_y >= 0) & (out_y < H_out)
                if not valid_mask.any():
                    continue

                y_overlap = torch.clamp(torch.minimum(out_y, y2) - torch.maximum(out_y - 1.0, y1), min=0.0)
                x_overlap = torch.clamp(torch.minimum(out_x, x2) - torch.maximum(out_x - 1.0, x1), min=0.0)
                weights = x_overlap * y_overlap * valid_mask.float()

                active_indices = weights > 0
                if not active_indices.any():
                    continue

                w_active = weights[active_indices]
                val_active = img_tensor[active_indices] * w_active
                
                c_idx = channels_flat[active_indices]
                y_idx = out_y[active_indices].long()
                x_idx = out_x[active_indices].long()
                
                # Calcul de l'index unique dans la structure globale de sortie [C, H_out, W_out]
                flat_target_indices = c_idx * (H_out * W_out) + y_idx * W_out + x_idx

                # Accumulation atomique robuste
                output_accum.view(-1).scatter_add_(0, flat_target_indices, val_active)
                weight_accum.view(-1).scatter_add_(0, flat_target_indices, w_active)

    print("Normalisation finale du cube couleur PyTorch...")
    final_stack = torch.where(weight_accum > 0, output_accum / weight_accum, 0.0)
    
    return final_stack.cpu().numpy(), weight_accum.cpu().numpy()
def gpu_bayer_drizzle_stack_torch(fits_paths, scale=2.0, pixfrac=0.6):
    """
    Empilement Drizzle PyTorch universel [C, H, W] prenant en compte 
    les pixels de référence (CRPIX1 / CRPIX2) pour éliminer les distorsions de rotation.
    """
    device = get_torch_device()
    print(f"Périphérique de calcul activé : {device}")

    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # 1. Utiliser le premier fichier pour définir les dimensions de SORTIE de référence
    with fits.open(fits_paths[0]) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        C, H_ref, W_ref = hdu.data.shape
        
        # Récupération du pixel de référence mondial/astrométrique de l'image maîtresse
        # En FITS, l'indexation commence à 1, donc on passe en indexation 0 avec -1
        crpix1_ref = float(hdu.header.get('CRPIX1', W_ref / 2.0)) - 1.0
        crpix2_ref = float(hdu.header.get('CRPIX2', H_ref / 2.0)) - 1.0

    # Les dimensions de sortie et son centre de référence sont mis à l'échelle
    H_out, W_out = int(H_ref * scale), int(W_ref * scale)
    crpix1_out = crpix1_ref * scale
    crpix2_out = crpix2_ref * scale

    output_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((C, H_out, W_out), dtype=torch.float32, device=device)

    r = (pixfrac * scale) / 2.0

    # 2. Boucle sur les fichiers FITS
    for i, path in enumerate(fits_paths):
        with fits.open(path) as hdul:
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            img = hdu.data
            header = hdu.header
            
            C_curr, H_in, W_in = img.shape
            
            # Paramètres de transformation Siril
            dx = float(header.get('DX', 0.0))
            dy = float(header.get('DY', 0.0))
            angle_deg = float(header.get('ANGLE', 0.0))
            angle_rad = np.radians(angle_deg)
            
            # Lecture du pixel de référence propre à l'image courante
            crpix1_curr = float(header.get('CRPIX1', W_in / 2.0)) - 1.0
            crpix2_curr = float(header.get('CRPIX2', H_in / 2.0)) - 1.0
            
            print(f"[{i+1}/{len(fits_paths)}] PyTorch -> {path} | Ref Pixel: ({crpix1_curr:.1f}, {crpix2_curr:.1f}) | DX: {dx:.2f}, DY: {dy:.2f}")

            img_tensor = torch.tensor(img, dtype=torch.float32, device=device)

        # Génération de la grille 2D
        y_in, x_in = torch.meshgrid(
            torch.arange(H_in, dtype=torch.float32, device=device),
            torch.arange(W_in, dtype=torch.float32, device=device),
            indexing='ij'
        )
        x_in_flat = x_in.reshape(-1)
        y_in_flat = y_in.reshape(-1)

        # --- CALCUL GÉOMÉTRIQUE CORRIGÉ ---
        # On centre les coordonnées par rapport au PIXEL DE RÉFÉRENCE réel, pas le centre géométrique
        xc_in = x_in_flat - crpix1_curr
        yc_in = y_in_flat - crpix2_curr

        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)

        # Application de la rotation autour du point pivot + translation
        xc_out = (xc_in * cos_t - yc_in * sin_t + dx) * scale
        yc_out = (xc_in * sin_t + yc_in * cos_t + dy) * scale

        # Repositionnement dans le repère de sortie basé sur le pixel de référence de sortie
        target_x = xc_out + crpix1_out
        target_y = yc_out + crpix2_out

        x1, x2 = target_x - r, target_x + r
        y1, y2 = target_y - r, target_y + r

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

                # On n'injecte que là où le poids est positif
                active_indices = weights > 0
                if not active_indices.any():
                    continue

                # --- AJOUT DU FILTRE POUR IGNORER LES PIXELS NOIRS (Siril style) ---
                # On récupère la valeur brute du pixel d'entrée pour les indices actifs
                # Si le pixel vaut 0.0 (ou moins), on l'exclut du calcul
                for channel in range(C_curr):
                    flat_channel_input = img_tensor[channel].reshape(-1)
                    
                    # On crée un sous-masque combinant la géométrie ET la valeur non nulle
                    valid_pixel_mask = active_indices.clone()
                    valid_pixel_mask[active_indices] = (flat_channel_input[active_indices] > 0.0)
                    
                    if not valid_pixel_mask.any():
                        continue
                        
                    w_active = weights[valid_pixel_mask]
                    val_active = flat_channel_input[valid_pixel_mask] * w_active
                    
                    y_idx = out_y[valid_pixel_mask].long()
                    x_idx = out_x[valid_pixel_mask].long()
                    flat_spatial_indices = y_idx * W_out + x_idx

                    # Accumulation uniquement des pixels utiles
                    output_accum[channel].view(-1).scatter_add_(0, flat_spatial_indices, val_active)
                    weight_accum[channel].view(-1).scatter_add_(0, flat_spatial_indices, w_active)

    print("Normalisation finale géométrique...")
    final_stack = torch.where(weight_accum > 0, output_accum / weight_accum, 0.0)
    
    return final_stack.cpu().numpy(), weight_accum.cpu().numpy()

# Exemple d'appel identique
if __name__ == "__main__":
    mes_images_raw_siril = [f"r_bkg_pp_lights_{i:05d}.fit.fz" for i in range(1, 54)]
    
    # Exécution
    image_rvb_mac, carte_poids = gpu_bayer_drizzle_stack_torch(mes_images_raw_siril, scale=2.0, pixfrac=0.6)
    
    fits.writeto("image_drizzle_torch.fits", image_rvb_mac, overwrite=True)
    print("Terminé ! Fichier créé sous 'image_drizzle_torch.fits'.")
