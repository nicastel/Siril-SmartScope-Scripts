import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sirilpy as s

s.ensure_installed("astropy")
from astropy.io import fits

s.ensure_installed("tqdm")
from tqdm import tqdm

# =====================================================================
# 1. COMPOSANTS DE L'ARCHITECTURE (BACKBONE & MODELES PHYSIQUES)
# =====================================================================

class AstroDIPBackbone(nn.Module):
    """CNN backbone parameterizing the clean, high-fidelity latent image z."""
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super(AstroDIPBackbone, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters + base_filters, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        b  = self.bottleneck(s2)
        d2 = self.dec2(b)
        return self.dec1(torch.cat([d2, s1], dim=1))


class GlobalBackgroundGradientModel(nn.Module):
    """Modélise un gradient de fond de ciel global basse fréquence via un polynôme 2D."""
    def __init__(self, degree=2):
        super(GlobalBackgroundGradientModel, self).__init__()
        self.degree = degree
        num_coefficients = (degree + 1) * (degree + 2) // 2
        self.coefficients = nn.Parameter(torch.zeros(num_coefficients))

    def forward(self, H, W, device):
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        background = torch.zeros((H, W), device=device)
        idx = 0
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                background += self.coefficients[idx] * (x_grid ** j) * (y_grid ** i)
                idx += 1
        return background.unsqueeze(0).unsqueeze(0)


class AstroSRDitheringObservationModel(nn.Module):
    """Modèle physique incluant Dithering, PSF HR, Fond de ciel et Downsampling x2."""
    def __init__(self, psf_kernels_hr, scale_factor=2, bg_degree=2):
        super(AstroSRDitheringObservationModel, self).__init__()
        self.register_buffer('psfs_hr', psf_kernels_hr)
        self.num_frames = psf_kernels_hr.shape[0]
        self.scale_factor = scale_factor
        self.bg_model_hr = GlobalBackgroundGradientModel(degree=bg_degree)
        self.shifts = nn.Parameter(torch.zeros(self.num_frames, 2)) 

    def _apply_shift(self, image_hr, dx, dy):
        N, C, H, W = image_hr.shape
        theta = torch.zeros((1, 2, 3), device=image_hr.device, dtype=image_hr.dtype)
        theta[:, 0, 0], theta[:, 1, 1] = 1.0, 1.0
        theta[:, 0, 2] = -2.0 * dx / W
        theta[:, 1, 2] = -2.0 * dy / H
        grid = F.affine_grid(theta, image_hr.size(), align_corners=False)
        return F.grid_sample(image_hr, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def fft_convolve(self, image, kernel):
        img_fft = torch.fft.rfft2(image, s=image.shape[-2:])
        kernel_fft = torch.fft.rfft2(kernel, s=image.shape[-2:])
        return torch.fft.irfft2(img_fft * kernel_fft, s=image.shape[-2:])

    def get_normalized_psfs(self):
        return self.psfs_hr # Dans cette configuration stable, les PSF d'ancrages restent fixes par tuile

    def forward(self, latent_z_hr):
        H_hr, W_hr = latent_z_hr.shape[-2:]
        device = latent_z_hr.device
        bg_surface_hr = self.bg_model_hr(H_hr, W_hr, device)
        degraded_frames_lr = []
        
        for i in range(self.num_frames):
            dx, dy = self.shifts[i, 0], self.shifts[i, 1]
            shifted_z_hr = self._apply_shift(latent_z_hr, dx, dy)
            
            psf_i_hr = self.psfs_hr[i:i+1]
            padded_psf = F.pad(psf_i_hr, (0, W_hr - psf_i_hr.shape[-1], 0, H_hr - psf_i_hr.shape[-2]))
            padded_psf = torch.roll(padded_psf, shifts=(-(psf_i_hr.shape[-1]//2), -(psf_i_hr.shape[-2]//2)), dims=(-2, -1))
            
            blurred_hr = self.fft_convolve(shifted_z_hr, padded_psf) + bg_surface_hr
            blurred_lr = F.interpolate(blurred_hr, scale_factor=1.0 / self.scale_factor, mode='area')
            degraded_frames_lr.append(blurred_lr)
            
        return torch.cat(degraded_frames_lr, dim=0), bg_surface_hr

# =====================================================================
# 2. FONCTIONS DE PERTE ET RECONNAISSANCE D'ÉTOILES
# =====================================================================

class AstroFWHMEstimator(nn.Module):
    """
    Estime rapidement la FWHM (Full Width at Half Maximum) moyenne
    des étoiles détectées sur l'image haute résolution (HR) pour le Early Stopping.
    """
    def __init__(self):
        super(AstroFWHMEstimator, self).__init__()

    @torch.no_grad()
    def forward(self, latent_z_hr, star_mask_lr, scale_factor=2):
        # 1. Isolation de la matrice 2D Haute Résolution propre
        height_hr, width_hr = latent_z_hr.shape[-2], latent_z_hr.shape[-1]
        stars_2d = torch.mean(latent_z_hr.view(-1, height_hr, width_hr), dim=0).clone()
        
        # Masque de garde pour éliminer les artefacts de bords de tuiles
        guard_mask = torch.zeros_like(stars_2d)
        guard_mask[30:-30, 30:-30] = 1.0
        stars_masked = stars_2d * guard_mask
        
        # 2. Localisation de la vraie étoile la plus brillante
        max_val = torch.max(stars_masked)
        if max_val < 1e-4:
            return 12.0
            
        idx_max = torch.argmax(stars_masked)
        peak_y = int(idx_max // width_hr)
        peak_x = int(idx_max % width_hr)
        
        # 3. Extraction d'une petite vignette étroite (15x15 pixels)
        radius = 7
        y_min = max(0, peak_y - radius)
        y_max = min(height_hr, peak_y + radius + 1)
        x_min = max(0, peak_x - radius)
        x_max = min(width_hr, peak_x + radius + 1)
        
        vignette = stars_2d[y_min:y_max, x_min:x_max]
        
        # 4. Grille de coordonnées locales
        ny, nx = vignette.shape
        y_grid, x_grid = torch.meshgrid(torch.arange(ny, device=vignette.device), 
                                        torch.arange(nx, device=vignette.device), indexing='ij')
        
        # --- CORRECTION DE LA DÉRIVE ---
        # Soustraction du fond local
        vignette_sub = torch.clamp(vignette - torch.min(vignette), min=0.0)
        
        # Seuil strict à mi-hauteur (Half-Maximum) local à l'étoile
        # Tout ce qui est en dessous de 50% de l'intensité du pic est mis à 0.
        # Cela coupe mathématiquement les fuites de pixels sur les bords du carré de 15x15
        seuil_hm = torch.max(vignette_sub) * 0.5
        vignette_clean = torch.where(vignette_sub >= seuil_hm, vignette_sub, torch.zeros_like(vignette_sub))
        
        total_flux = torch.sum(vignette_clean) + 1e-8
        
        # 5. Calcul des moments sur l'étoile purement isolée
        local_cy = torch.sum(y_grid * vignette_clean) / total_flux
        local_cx = torch.sum(x_grid * vignette_clean) / total_flux
        
        var_y = torch.sum(((y_grid - local_cy) ** 2) * vignette_clean) / total_flux
        var_x = torch.sum(((x_grid - local_cx) ** 2) * vignette_clean) / total_flux
        
        # Conversion Variance -> FWHM (FWHM = 2.355 * sigma)
        sigma = torch.sqrt((var_y + var_x) / 2.0)
        fwhm_estimate = 2.355 * sigma.item()
        
        print(f" [DEBUG FWHM] Mesure locale sur le pic central : {fwhm_estimate:.4f} px")
        
        if fwhm_estimate < 1.0 or fwhm_estimate > 20.0:
            return 12.0
            
        return fwhm_estimate

class AstroMixedNoiseLoss(nn.Module):
    """
    Version MSE simplifiée pour stabiliser les gradients 
    et empêcher le blocage de la Loss en production.
    """
    def __init__(self, gain=1.0, read_noise=0.0):
        super(AstroMixedNoiseLoss, self).__init__()
        # Ces variables restent déclarées pour ne pas casser le reste du script
        self.gain = gain
        self.read_noise = read_noise

    def forward(self, predicted_exposures, observed_exposures):
        # Calcul direct de l'erreur quadratique moyenne (MSE)
        # Entièrement stable, linéaire et convexe pour le réseau de neurones
        return F.mse_loss(predicted_exposures, observed_exposures)


class AstroDynamicInvalidationLoss(nn.Module):
    def __init__(self, base_criterion, sigma_thresh=4.5, start_iter=300):
        super(AstroDynamicInvalidationLoss, self).__init__()
        self.base_criterion = base_criterion
        self.sigma_thresh = sigma_thresh
        self.start_iter = start_iter

    def forward(self, predicted_exposures, observed_exposures, current_step):
        with torch.no_grad():
            residuals = torch.abs(predicted_exposures - observed_exposures)
            std_per_frame = torch.std(residuals, dim=(-2, -1), keepdim=True)
            if current_step >= self.start_iter:
                mask = (residuals < (self.sigma_thresh * std_per_frame)).float()
            else:
                mask = torch.ones_like(observed_exposures)
        
        raw_loss = self.base_criterion(predicted_exposures, observed_exposures)
        return torch.sum(raw_loss * mask) / (torch.sum(mask) + 1e-8)


class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1e-5):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return self.weight * (torch.sum(diff_h) + torch.sum(diff_w))


class AstroPhotometryConservationLoss(nn.Module):
    def __init__(self, weight=1e-3, patch_size=32):
        super(AstroPhotometryConservationLoss, self).__init__()
        self.weight = weight
        self.patch_size = patch_size

    def forward(self, latent_z, observed_exposures):
        # 1. Extraction de la géométrie réelle de la tuile
        height_lr, width_lr = observed_exposures.shape[-2], observed_exposures.shape[-1]
        height_hr, width_hr = latent_z.shape[-2], latent_z.shape[-1]
        
        # Déduction dynamique du facteur de zoom (ex: 1024 / 512 = 2)
        scale_factor = height_hr // height_lr
        
        # 2. Réduction stricte du tenseur observé en 4D [1, 1, H_lr, W_lr]
        # On écrase tous les axes supérieurs (batch, canal, frames) pour obtenir une image moyenne plane
        flattened_frames = observed_exposures.view(-1, height_lr, width_lr)
        mean_frame_2d = torch.mean(flattened_frames, dim=0, keepdim=False)
        mean_observed_frame = mean_frame_2d.unsqueeze(0).unsqueeze(0).to(latent_z.device)
        
        # 3. Contrainte de Flux Globale (Somme totale)
        global_loss = F.mse_loss(torch.sum(latent_z), torch.sum(mean_observed_frame))
        
        # 4. Ajustement géométrique des patchs pour compenser la Super-Résolution
        # Le pool HR utilise un kernel plus grand pour correspondre à la taille physique du pool LR
        patch_size_hr = self.patch_size * scale_factor
        
        local_flux_latent = F.avg_pool2d(latent_z, kernel_size=patch_size_hr, stride=patch_size_hr)
        local_flux_observed = F.avg_pool2d(mean_observed_frame, kernel_size=self.patch_size, stride=self.patch_size)
        
        # 5. Calcul final de la perte : les deux tenseurs font désormais strictement [1, 1, H_patch, W_patch]
        return self.weight * (global_loss + F.mse_loss(local_flux_latent, local_flux_observed))


class AstroSparsityL1Loss(nn.Module):
    def __init__(self, weight_pixel=1e-6, weight_gradient=1e-6):
        super(AstroSparsityL1Loss, self).__init__()
        self.weight_pixel = weight_pixel
        self.weight_gradient = weight_gradient

    def forward(self, latent_z):
        loss_l1_pixel = torch.mean(torch.abs(latent_z))
        grad_x = latent_z[:, :, :, 1:] - latent_z[:, :, :, :-1]
        grad_y = latent_z[:, :, 1:, :] - latent_z[:, :, :-1, :]
        loss_l1_grad = torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))
        return (self.weight_pixel * loss_l1_pixel) + (self.weight_gradient * loss_l1_grad)


class PolynomialL2Regularization(nn.Module):
    def __init__(self, weight=1e-3):
        super(PolynomialL2Regularization, self).__init__()
        self.weight = weight

    def forward(self, coefficients):
        return self.weight * torch.sum(coefficients[1:] ** 2)


class AstroStarFinder(nn.Module):
    """
    Star Finder ultra-robuste basé sur le gradient de Sobel et une
    normalisation interne stricte. Ignore le bruit de fond de ciel continu.
    """
    def __init__(self, sigma_thresh=8.0, min_star_pixels=15):
        super(AstroStarFinder, self).__init__()
        self.sigma_thresh = sigma_thresh
        self.min_star_pixels = min_star_pixels
        
        # Filtres de Sobel pour détecter les vraies structures (les bords d'étoiles)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    @torch.no_grad()
    def forward(self, tile_lr):
        # 1. Extraction et normalisation géométrique plane [1, 1, H, W]
        height, width = tile_lr.shape[-2], tile_lr.shape[-1]
        mean_frame = torch.mean(tile_lr.view(-1, height, width), dim=0).unsqueeze(0).unsqueeze(0)
        
        # --- FILTRE COMPLÉMENTAIRE DE BRUIT STRUCTURÉ (FLUX ABSOLU) ---
        # Si l'écart de dynamique de la tuile entière est infime, c'est du pur fond de ciel continu
        v_min, v_max = torch.min(mean_frame), torch.max(mean_frame)
        dynamique = v_max - v_min
        
        # Normalisation interne
        if dynamique > 1e-5:
            mean_frame = (mean_frame - v_min) / dynamique
            
        # 2. Calcul des gradients spatiaux (Sobel)
        grad_x = F.conv2d(mean_frame, self.sobel_x, padding=1)
        grad_y = F.conv2d(mean_frame, self.sobel_y, padding=1)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 3. Seuil statistique robuste
        median = torch.median(magnitude)
        mad = torch.median(torch.abs(magnitude - median))
        sigma = 1.4826 * mad
        
        star_mask = (magnitude > (median + self.sigma_thresh * sigma)).float()
        star_mask = F.max_pool2d(star_mask, kernel_size=3, stride=1, padding=1) * star_mask
        
        num_pixels = torch.sum(star_mask).item()
        
        # --- PROTECTION FINALE CONTRE LE BRUIT DE FOND STRUCTURÉ ---
        # Si le nombre de pixels est suspect ET que la dynamique absolue est typique d'un fond de ciel,
        # ou pour forcer le basculement si vous savez que le fond est homogène :
        # On force has_enough_stars à False si la densité est inférieure à 1% ou si l'intensité max est basse
        star_density = num_pixels / star_mask.numel()
        
        # On durcit le critère : il faut au moins un minimum de contraste local 
        # pour valider qu'il s'agit de vraies étoiles et non d'une trame de bruit
        has_enough_stars = (num_pixels >= self.min_star_pixels) and (star_density > 0.008)
        
        # Si vous voulez tester DIRECTEMENT le comportement en mode VIDE sur cette tuile,
        # vous pouvez temporairement forcer : has_enough_stars = False
        
        return star_density, has_enough_stars, star_mask

# =====================================================================
# 3. INTERPOLATEUR DE PSF SPATIALE (EXEMPLE CONFIGURABLE MOFFAT)
# ====================================================================
class SpatialMoffatInterpolator:
    """Génère un cube de PSF locales interpolées selon la position sur le capteur."""
    def __init__(self, num_frames, image_shape, kernel_size=31):
        self.num_frames = num_frames
        self.img_h, self.img_w = image_shape
        self.kernel_size = kernel_size
        center = kernel_size // 2
        y, x = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing='ij')
        self.r_squared = (x - center)**2 + (y - center)**2

    def __call__(self, y_center, x_center):
        cy, cx = self.img_h / 2, self.img_w / 2
        dist = math.sqrt((y_center - cy)**2 + (x_center - cx)**2) / math.sqrt(cy**2 + cx**2)
        alpha_local = 2.5 + 3.0 * dist  # Simulation d'évasement optique en bordure
        psf = (1.0 + (self.r_squared / (alpha_local**2))) ** (-2.5)
        psf = psf / torch.sum(psf)
        return psf.unsqueeze(0).unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

    
# =====================================================================
# 4. BOUCLE D'OPTIMISATION DE TUILE UNIQUE
# =====================================================================
def optimize_astro_tile_batched(observed_exposures_lr, psf_kernels_hr, scale_factor=2, 
                                batch_size=16, iterations=1000, **kwargs):
    """
    Version optimisée pour la VRAM avec mini-batch temporel.
    batch_size: Nombre maximal de fichiers envoyés simultanément au GPU (ex: 16 ou 32).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # 1. On nettoie les dimensions pour obtenir la forme brute [Num_Frames, 1, H, W]
    if observed_exposures_lr.dim() == 5:
        observed_exposures_lr = observed_exposures_lr.squeeze(0).squeeze(0)
    if observed_exposures_lr.dim() == 3:
        observed_exposures_lr = observed_exposures_lr.unsqueeze(1)
        
    num_total_frames = observed_exposures_lr.shape[0]
    H_lr, W_lr = observed_exposures_lr.shape[-2:]
    H_hr, W_hr = H_lr * scale_factor, W_lr * scale_factor
    
    # 2. Initialisation des modèles sur le GPU
    net = AstroDIPBackbone().to(device)
    # Note: On initialise le modèle d'observation complet
    forward_model = AstroSRDitheringObservationModel(psf_kernels_hr, scale_factor, kwargs.get('bg_degree', 2)).to(device)
    fixed_noise_input_hr = torch.randn(1, 1, H_hr, W_hr, device=device) * 0.1
    
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 0.01},
        {'params': forward_model.bg_model_hr.parameters(), 'lr': 0.005},
        {'params': forward_model.shifts, 'lr': 0.02}
    ])
    
    # Criteres de perte
    base_data_loss = AstroMixedNoiseLoss(gain=kwargs.get('gain', 2.1), read_noise=kwargs.get('read_noise', 4.5))
    dni_data_criterion = AstroDynamicInvalidationLoss(base_criterion=base_data_loss, start_iter=300)
    tv_criterion = TotalVariationLoss(weight=kwargs.get('tv_weight', 1e-5))
    photo_criterion = AstroPhotometryConservationLoss(weight=kwargs.get('photo_weight', 1e-3), patch_size=32 * scale_factor)
    sparsity_criterion = AstroSparsityL1Loss(weight_pixel=kwargs.get('l1_weight', 1e-6), weight_gradient=1e-6)
    bg_l2_criterion = PolynomialL2Regularization(weight=kwargs.get('bg_l2_weight', 1e-3))
    
    from tqdm import tqdm
    progress_bar = tqdm(range(iterations), desc="   ↳ Itérations DIP (Batched)", leave=False)
    
    # --- Remplacer la boucle d'optimisation dans votre fonction optimize_astro_tile_batched ---

    fwhm_estimator = AstroFWHMEstimator()
    
    # Paramètres de Early Stopping
    meilleure_fwhm = float('inf')
    patience = 50
    declenchements_sans_amelioration = 0
    iteration_arret = iterations
    
    # Récupération du masque d'étoiles calculé en amont pour cette tuile (passé via kwargs)
    star_mask_lr = kwargs.get('tile_mask', torch.ones((1, 1, H_lr, W_lr)))

    fwhm_estimator = AstroFWHMEstimator()
    
    # --- CALCUL DE LA FWHM INITIALE DE RÉFÉRENCE ---
    # On mesure la FWHM sur la moyenne des images brutes observées (LR)
    mean_obs_lr = torch.mean(observed_exposures_lr, dim=0, keepdim=True)
    # On l'estime à l'échelle LR
    fwhm_initiale_lr = fwhm_estimator(mean_obs_lr, star_mask_lr, scale_factor=1)
    
    # Comme le réseau z travaille à l'échelle HR (x2), la FWHM cible équivalente 
    # sur l'image brute extrapolée serait :
    fwhm_reference_hr = fwhm_initiale_lr * scale_factor
    
    print(f"   ↳ 🔍 FWHM brute initiale (échelle LR) : {fwhm_initiale_lr:.2f}px")
    print(f"   ↳ 🎯 FWHM cible maximale (échelle HR) : {fwhm_reference_hr:.2f}px (Le traitement doit descendre sous ce seuil)")

    # Paramètres de Early Stopping
    meilleure_fwhm = float('inf')
    patience = 50
    declenchements_sans_amelioration = 0
    iteration_arret = iterations

    app = s.SirilInterface()
    try:
        app.connect()
        print("Connected successfully!")
    except SirilConnectionError as e:
        print(f"Connection failed: {e}")

    for step in progress_bar:
        optimizer.zero_grad()
        
        # Sélection du batch temporel
        if num_total_frames > batch_size:
            indices_batch = torch.randperm(num_total_frames)[:batch_size]
        else:
            indices_batch = torch.arange(num_total_frames)
            
        # --- Extraction locale du batch ---
        batch_obs_lr = observed_exposures_lr[indices_batch].to(device)
        
        # CORRECTION DU USERWARNING (Ligne 215) :
        # Si batch_obs_lr a 5 dimensions, on fusionne le batch 
        # et les frames pour obtenir un tenseur 4D strict compatible avec predicted_lr
        if batch_obs_lr.dim() == 5:
            # On supprime la dimension d'index 1 superflue s'il s'agit d'un canal unitaire
            batch_obs_lr = batch_obs_lr.squeeze(1) # Reste [5, 3, 512, 512]
            # On extrait les dimensions
            b_sz, n_fr, h_lr, w_lr = batch_obs_lr.shape
            # On change la forme en combinant les axes pour correspondre aux 5 prédictions de predicted_lr
            # predicted_lr faisant, si vos PSF du modèle génèrent 5 sorties, 
            # il faut s'assurer que target possède exactement la même forme de batch.
            # Si le modèle renvoie une seule image par élément du batch, on prend la moyenne des 3 frames :
            batch_obs_lr = torch.mean(batch_obs_lr, dim=1, keepdim=True) # Devient [5, 1, 512, 512]

        latent_z_hr = net(fixed_noise_input_hr)
        
        # Sauvegarde et swap des buffers pour le forward model
        backup_psfs, backup_shifts = forward_model.psfs_hr, forward_model.shifts
        forward_model.psfs_hr = forward_model.psfs_hr[indices_batch]
        forward_model.shifts = nn.Parameter(forward_model.shifts[indices_batch])
        forward_model.num_frames = len(indices_batch)
        
        predicted_lr, bg_hr = forward_model(latent_z_hr)
        
        # Pertes
        loss_data = dni_data_criterion(predicted_lr, batch_obs_lr, step)
        loss_tv = tv_criterion(latent_z_hr)
        loss_photo = photo_criterion(latent_z_hr, batch_obs_lr)
        loss_l1 = sparsity_criterion(latent_z_hr)
        loss_bg_l2 = bg_l2_criterion(forward_model.bg_model_hr.coefficients)
        loss_anchor = kwargs.get('psf_anchor_weight', 1.0) * torch.mean(forward_model.shifts ** 2)
        
        total_loss = loss_data + loss_tv + loss_photo + loss_l1 + loss_bg_l2 + loss_anchor
        total_loss.backward()
        
        # Restauration des gradients
        with torch.no_grad():
            backup_shifts.grad = torch.zeros_like(backup_shifts)
            backup_shifts.grad[indices_batch] = forward_model.shifts.grad
            
        forward_model.psfs_hr, forward_model.shifts, forward_model.num_frames = backup_psfs, backup_shifts, num_total_frames
        optimizer.step()

        # --- EXPORT ET CHARGEMENT AUTOMATIQUE DANS SIRIL ---
        if step % 50 == 0:
            with torch.no_grad():
                preview_sky_hr = net(fixed_noise_input_hr).detach().cpu().squeeze().numpy()
                hdu_preview = fits.PrimaryHDU(data=preview_sky_hr)
                
                # Sauvegarde du fichier temporaire
                nom_fichier_temp = "clearnet_live.fits"
                hdu_preview.writeto(nom_fichier_temp, overwrite=True)
                
                # ENVOI DE LA COMMANDE À SIRIL VIA PYSIRIL
                # Si vous avez instancié Siril avec 'app = Siril()', vous l'appelez ainsi :
                try:
                    # La commande 'load' force l'interface de Siril à afficher le nouveau FITS
                    app.cmd(f"load {nom_fichier_temp}")
                except NameError:
                    # Sécurité si 'app' n'est pas définie dans cette sous-fonction
                    pass
        
        with torch.no_grad():
            forward_model.shifts.data[0, :] = 0.0
            
        # --- CALCUL ET LOGIQUE DU EARLY STOPPING ---
        # --- CALCUL ET LOGIQUE DU EARLY STOPPING ---
        if step >= 150:
            fwhm_actuelle = fwhm_estimator(latent_z_hr, star_mask_lr.to(device), scale_factor)
            
            # On vérifie si l'image actuelle fait MIEUX que la meilleure itération
            if fwhm_actuelle < meilleure_fwhm:
                meilleure_fwhm = fwhm_actuelle
                declenchements_sans_amelioration = 0
            else:
                declenchements_sans_amelioration += 1
                
            # --- AJOUT DE LA VÉRIFICATION D'EFFET ---
            # On calcule le gain de netteté en % par rapport aux images brutes
            gain_nettete = ((fwhm_reference_hr - fwhm_actuelle) / fwhm_reference_hr) * 100.0
            
            # Forcer la mise à jour textuelle dans tqdm
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "FWHM_HR": f"{fwhm_actuelle:.2f}px",
                "Gain": f"{gain_nettete:.1f}%",
                "Patience": f"{declenchements_sans_amelioration}/{patience}"
            })
            
            # Sécurité d'arrêt : on ne coupe que si la FWHM stagne ET qu'on a bien
            # obtenu une amélioration (Gain > 0) par rapport aux fichiers FITS bruts
            if declenchements_sans_amelioration >= patience and gain_nettete > 0.0 and kwargs.get('has_stars', True):
                iteration_arret = step
                break
        elif step % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{total_loss.item():.4f}", "FWHM_HR": "Calcul..."})

    print(f"   ↳ 🏁 Fin de la tuile à l'itération {iteration_arret}/{iterations} | Meilleure FWHM HR : {meilleure_fwhm:.2f}px")

    with torch.no_grad():
        final_sky_hr = net(fixed_noise_input_hr)
        _, final_bg_hr = forward_model(final_sky_hr)
        
    return final_sky_hr.detach().cpu(), final_bg_hr.detach().cpu()


# =====================================================================
# 5. GESTIONNAIRE DE TRAITEMENT GLOBAL PAR TUILES (TILING PIPELINE)
# =====================================================================
def run_astro_clearnet_pipeline(fits_paths, output_prefix="output", tile_size=512, overlap=64, iterations=1000):
    """Charge l'ensemble des fichiers FITS, applique le découpage par tuiles et exporte les résultats."""
    print(f"📦 Chargement de {len(fits_paths)} fichiers FITS...")
    
    # --- À REMPLACER DANS run_astro_clearnet_pipeline (Chargement des FITS) ---
    frames = []
    base_header = None
    for path in fits_paths:
        with fits.open(path, memmap=True) as hdul:
            data = hdul[0].data.astype(np.float32)
            if base_header is None:
                base_header = hdul[0].header
            frames.append(data)
            
    # Empilement des images
    observed_exposures = torch.tensor(np.stack(frames)).unsqueeze(1)
    
    # CORRECTION CRITIQUE : Normalisation globale de la dynamique entre 0.0 et 1.0
    # Empêche la saturation de la Sigmoid et débloque la descente de gradient
    valeur_max_pixel = torch.max(observed_exposures)
    if valeur_max_pixel > 1.0:
        print(f"⚠️ Pixel max détecté à {valeur_max_pixel.item():.1f} ADU. Normalisation automatique [0, 1] en cours...")
        observed_exposures = observed_exposures / valeur_max_pixel

    N, O, C, H_lr, W_lr = observed_exposures.shape
    
    psf_interpolator = SpatialMoffatInterpolator(num_frames=N, image_shape=(H_lr, W_lr))
    # 1. Dans l'initialisation de run_astro_clearnet_pipeline, montez le seuil à 8.0
    star_finder = AstroStarFinder(sigma_thresh=8.0, min_star_pixels=15)
        
    scale_factor = 2
    H_hr, W_hr = H_lr * scale_factor, W_lr * scale_factor
    global_sky_hr = torch.zeros((1, 1, H_hr, W_hr))
    global_bg_hr = torch.zeros((1, 1, H_hr, W_hr))
    global_mask_lr = torch.zeros((H_lr, W_lr))
    weight_accumulator_hr = torch.zeros((1, 1, H_hr, W_hr))
    
    tile_size_hr = tile_size * scale_factor
    overlap_hr = overlap * scale_factor
    w_tile_hr = torch.ones((tile_size_hr, tile_size_hr))
    for i in range(overlap_hr):
        val = 0.5 - 0.5 * math.cos(math.pi * i / overlap_hr)
        w_tile_hr[i, :] *= val
        w_tile_hr[-1-i, :] *= val
        w_tile_hr[:, i] *= val
        w_tile_hr[:, -1-i] *= val

    stride_lr = tile_size - overlap

 # 1. Calcul préalable du nombre total de tuiles pour calibrer le compteur
    steps_y = list(range(0, H_lr, stride_lr))
    steps_x = list(range(0, W_lr, stride_lr))
    total_tuiles = len(steps_y) * len(steps_x)
    
    print(f"🧩 Découpage de l'image en {total_tuiles} tuiles...")
    
    # 2. Création de la barre de progression principale
    global_progress = tqdm(total=total_tuiles, desc="🚀 Progression AstroClearNet")

    for y in steps_y:
        for x in steps_x:
            y_start = min(y, H_lr - tile_size)
            y_end = y_start + tile_size
            x_start = min(x, W_lr - tile_size)
            x_end = x_start + tile_size
            
            # Extraction de la tuile active
            tile_lr = observed_exposures[..., y_start:y_end, x_start:x_end]
            
            # CORRECTION CRITIQUE : On normalise la dynamique entre 0.0 et 1.0 D'ABORD
            valeur_max_tile = torch.max(tile_lr)
            if valeur_max_tile > 1.0:
                tile_lr = tile_lr / valeur_max_tile
                
            # Maintenant que la tuile est propre et normalisée, on lance le Star Finder
            density, has_stars, tile_mask = star_finder(tile_lr)
            
            # Enregistrement du masque binaire global
            global_mask_lr[y_start:y_end, x_start:x_end] = torch.max(
                global_mask_lr[y_start:y_end, x_start:x_end], tile_mask.squeeze()
            )
            
            # L'affichage du diagnostic reflétera enfin la réalité physique normalisée
            print("="*60)
            print(f"📊 [DIAGNOSTIC TUILE] Coordonnées LR : Y[{y_start}:{y_end}], X[{x_start}:{x_end}]")
            print(f"   ↳ Nombre de pixels détectés comme étoiles : {int(torch.sum(tile_mask).item())} px")
            print(f"   ↳ Densité stellaire calculée : {density*100:.4f} %")
            if has_stars:
                print(f"   ↳ 🟩 STATUT : RICHE EN ÉTOILES -> Optimisation Blind-PSF locale activée.")
                anchor_w = 1.0
            else:
                print(f"   ↳ 🟨 STATUT : VIDE / DIFFUSE -> Verrouillage de sécurité sur la PSF interpolée.")
                anchor_w = 1000.0
            print("="*60)
            
            y_center, x_center = y_start + (tile_size // 2), x_start + (tile_size // 2)
            psf_kernels_hr_local = psf_interpolator(y_center, x_center)
            
            anchor_w = 1.0 if has_stars else 1000.0
            
            # Optimisation de la tuile active (qui va afficher sa propre sous-barre d'itérations)
            tile_sky_hr, tile_bg_hr = optimize_astro_tile_batched(
                observed_exposures_lr=tile_lr, 
                psf_kernels_hr=psf_kernels_hr_local, 
                scale_factor=scale_factor,
                psf_anchor_weight=anchor_w, 
                iterations=iterations,
                batch_size=16,
                tile_mask=tile_mask,  # TRANSMISSION DU MASQUE UNIQUE DE CETTE TUILE
                has_stars=has_stars   # PERMET D'IGNORER LE EARLY STOPPING SUR LES TUILES VIDES
            )
            
            # ... [Logique d'accumulation inchangée] ...
            y_start_hr, y_end_hr = y_start * scale_factor, y_end * scale_factor
            x_start_hr, x_end_hr = x_start * scale_factor, x_end * scale_factor
            global_sky_hr[:, :, y_start_hr:y_end_hr, x_start_hr:x_end_hr] += tile_sky_hr * w_tile_hr
            global_bg_hr[:, :, y_start_hr:y_end_hr, x_start_hr:x_end_hr] += tile_bg_hr * w_tile_hr
            weight_accumulator_hr[:, :, y_start_hr:y_end_hr, x_start_hr:x_end_hr] += w_tile_hr
            
            # 3. Avancement d'un pas sur le compteur global à chaque tuile terminée
            global_progress.update(1)
            
    # Fermeture propre du compteur à la fin de la boucle
    global_progress.close()

    final_sky = (global_sky_hr / (weight_accumulator_hr + 1e-8)).squeeze().numpy()
    final_bg = (global_bg_hr / (weight_accumulator_hr + 1e-8)).squeeze().numpy()
    final_mask = global_mask_lr.numpy()
    
    print("💾 Enregistrement des fichiers FITS finaux...")
    
    hr_header = base_header.copy() if base_header else fits.Header()
    if 'CRPIX1' in hr_header: hr_header['CRPIX1'] *= scale_factor
    if 'CRPIX2' in hr_header: hr_header['CRPIX2'] *= scale_factor
    if 'CDELT1' in hr_header: hr_header['CDELT1'] /= scale_factor
    if 'CDELT2' in hr_header: hr_header['CDELT2'] /= scale_factor
    
    fits.writeto(f"{output_prefix}_clearnet_clean_x2.fits", final_sky, hr_header, overwrite=True)
    fits.writeto(f"{output_prefix}_clearnet_background_x2.fits", final_bg, hr_header, overwrite=True)
    fits.writeto(f"{output_prefix}_clearnet_star_mask.fits", final_mask.astype(np.int16), base_header, overwrite=True)
    
    print("✨ Opération terminée avec succès ! Les fichiers FITS ont été générés.")

def main():
    repertoire_donnees = "./"
    prefixe_sortie = "target_field"
    
    taille_tuile = 512
    chevauchement = 64
    nombre_iterations = 800  # Calé sur votre optimisation à 800 itérations
    
    print("🔭 --- DÉMARRAGE DU PIPELINE ASTROCLEARNET ---")
    
    # CORRECTION DU FILTRE : On exclut le préfixe de sortie ET le fichier live temporaire
    fichiers_cibles = [
        os.path.join(repertoire_donnees, f) 
        for f in os.listdir(repertoire_donnees) 
        if f.endswith('.fits') 
        and not f.startswith(prefixe_sortie) 
        and "clearnet_live" not in f  # <-- Exclusion stricte du fichier preview
    ]
    
    fichiers_cibles.sort()
    
    if len(fichiers_cibles) < 2:
        print(f"❌ Erreur : Il faut au moins 2 images FITS. Trouvées : {len(fichiers_cibles)}.")
        return

    print(f"✨ {len(fichiers_cibles)} expositions brutes prêtes pour le traitement.")
    
    try:
        run_astro_clearnet_pipeline(
            fits_paths=fichiers_cibles, 
            output_prefix=prefixe_sortie, 
            tile_size=taille_tuile, 
            overlap=chevauchement,
            iterations=nombre_iterations
        )
    except Exception as e:
        print(f"❌ Erreur critique : {str(e)}")
        raise e

if __name__ == "__main__":
    main()
