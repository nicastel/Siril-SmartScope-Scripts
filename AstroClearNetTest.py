import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits

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

class AstroMixedNoiseLoss(nn.Module):
    def __init__(self, gain=1.0, read_noise=0.0):
        super(AstroMixedNoiseLoss, self).__init__()
        self.gain = gain
        self.read_noise = read_noise

    def forward(self, predicted_exposures, observed_exposures):
        pred_electrons = predicted_exposures * self.gain
        obs_electrons = observed_exposures * self.gain
        variance = torch.clamp(pred_electrons, min=0.0) + (self.read_noise ** 2)
        return ((pred_electrons - obs_electrons) ** 2) / (variance + 1e-6)


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
        mean_observed_frame = torch.mean(observed_exposures, dim=0, keepdim=True)
        global_loss = F.mse_loss(torch.sum(latent_z), torch.sum(mean_observed_frame))
        local_flux_latent = F.avg_pool2d(latent_z, kernel_size=self.patch_size, stride=self.patch_size)
        local_flux_observed = F.avg_pool2d(mean_observed_frame, kernel_size=self.patch_size, stride=self.patch_size)
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
    """Star Finder rapide pour adapter dynamiquement la contrainte de la PSF."""
    def __init__(self, sigma_thresh=5.0, min_star_pixels=15):
        super(AstroStarFinder, self).__init__()
        self.sigma_thresh = sigma_thresh
        self.min_star_pixels = min_star_pixels
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', laplacian)

    @torch.no_grad()
    def forward(self, tile_lr):
        mean_frame = torch.mean(tile_lr, dim=0, keepdim=True)
        high_freq = F.conv2d(mean_frame, self.kernel, padding=1)
        sigma = 1.4826 * torch.median(torch.abs(high_freq - torch.median(high_freq)))
        star_mask = (high_freq > (torch.median(high_freq) + self.sigma_thresh * sigma)).float()
        num_pixels = torch.sum(star_mask).item()
        return num_pixels / star_mask.numel(), num_pixels >= self.min_star_pixels, star_mask

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
def optimize_astro_tile(observed_exposures_lr, psf_kernels_hr, scale_factor=2, 
                        gain=2.1, read_noise=4.5, tv_weight=1e-5, photo_weight=1e-3, 
                        l1_weight=1e-6, bg_degree=2, bg_l2_weight=1e-3, 
                        psf_anchor_weight=1.0, iterations=1000):
    """Optimise de manière auto-supervisée (DIP) une tuile de l'image globale."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    observed_exposures_lr = observed_exposures_lr.to(device)
    
    H_lr, W_lr = observed_exposures_lr.shape[-2:]
    H_hr, W_hr = H_lr * scale_factor, W_lr * scale_factor
    
    net = AstroDIPBackbone().to(device)
    forward_model = AstroSRDitheringObservationModel(psf_kernels_hr.to(device), scale_factor, bg_degree).to(device)
    fixed_noise_input_hr = torch.randn(1, 1, H_hr, W_hr, device=device) * 0.1
    
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 0.01},
        {'params': forward_model.bg_model_hr.parameters(), 'lr': 0.005},
        {'params': forward_model.shifts, 'lr': 0.02}
    ])
    
    base_data_loss = AstroMixedNoiseLoss(gain=gain, read_noise=read_noise)
    dni_data_criterion = AstroDynamicInvalidationLoss(base_criterion=base_data_loss, start_iter=300)
    tv_criterion = TotalVariationLoss(weight=tv_weight)
    photo_criterion = AstroPhotometryConservationLoss(weight=photo_weight, patch_size=32 * scale_factor)
    sparsity_criterion = AstroSparsityL1Loss(weight_pixel=l1_weight, weight_gradient=l1_weight)
    bg_l2_criterion = PolynomialL2Regularization(weight=bg_l2_weight)
    
    for step in range(iterations):
        optimizer.zero_grad()
        latent_z_hr = net(fixed_noise_input_hr)
        predicted_lr, bg_hr = forward_model(latent_z_hr)
        
        loss_data = dni_data_criterion(predicted_lr, observed_exposures_lr, step)
        loss_tv = tv_criterion(latent_z_hr)
        loss_photo = photo_criterion(latent_z_hr, observed_exposures_lr)
        loss_l1 = sparsity_criterion(latent_z_hr)
        loss_bg_l2 = bg_l2_criterion(forward_model.bg_model_hr.coefficients)
        loss_anchor = psf_anchor_weight * torch.mean(forward_model.shifts ** 2)
        
        total_loss = loss_data + loss_tv + loss_photo + loss_l1 + loss_bg_l2 + loss_anchor
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            forward_model.shifts.data[0, :] = 0.0 # Verrouillage strict de l'ancre
            
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
    
    frames = []
    base_header = None
    for path in fits_paths:
        with fits.open(path) as hdul:
            hdul.verify('fix')
            data = hdul[0].data.astype(np.float32)
            if base_header is None:
                base_header = hdul[0].header
            frames.append(data)
            
    observed_exposures = torch.tensor(np.stack(frames)).unsqueeze(1)
    print(observed_exposures.shape)
    N, O, C, H_lr, W_lr = observed_exposures.shape
    
    psf_interpolator = SpatialMoffatInterpolator(num_frames=N, image_shape=(H_lr, W_lr))
    star_finder = AstroStarFinder()
    
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

    for y in range(0, H_lr, stride_lr):
        for x in range(0, W_lr, stride_lr):
            y_start = min(y, H_lr - tile_size)
            x_start = min(x, W_lr - tile_size)
            y_end = y_start + tile_size
            x_end = x_start + tile_size
            
            tile_lr = observed_exposures[:, :, y_start:y_end, x_start:x_end]
            density, has_stars, tile_mask = star_finder(tile_lr)
            
            global_mask_lr[y_start:y_end, x_start:x_end] = torch.max(
                global_mask_lr[y_start:y_end, x_start:x_end], tile_mask.squeeze()
            )
            
            y_center, x_center = y_start + (tile_size // 2), x_start + (tile_size // 2)
            psf_kernels_hr_local = psf_interpolator(y_center, x_center)
            
            anchor_w = 1.0 if has_stars else 1000.0
            status_text = "Riche" if has_stars else "Vide/Diffuse (PSF Ancrée)"
            print(f"-> Traitement Tuile LR [{y_start}:{y_end}, {x_start}:{x_end}] | Statut: {status_text} | Densité: {density:.4f}")
            
            tile_sky_hr, tile_bg_hr = optimize_astro_tile(
                tile_lr, psf_kernels_hr_local, scale_factor=scale_factor,
                psf_anchor_weight=anchor_w, iterations=iterations
            )
            
            y_start_hr, y_end_hr = y_start * scale_factor, y_end * scale_factor
            x_start_hr, x_end_hr = x_start * scale_factor, x_end * scale_factor
            
            global_sky_hr[:, :, y_start_hr:y_end_hr, x_start_hr:x_end_hr] += tile_sky_hr * w_tile_hr
            global_bg_hr[:, :, y_start_hr:y_end_hr, x_start_hr:x_end_hr] += tile_bg_hr * w_tile_hr
            weight_accumulator_hr[:, :, y_start_hr:y_end_hr, x_start_hr:x_end_hr] += w_tile_hr

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
    """
    Fonction principale orchestrant l'exécution du pipeline AstroClearNet.
    Fouille le répertoire, prépare les variables et traite le lot d'images.
    """
    # 1. Configuration des répertoires et préfixes
    repertoire_donnees = "./"
    prefixe_sortie = "target_field"
    
    # Paramètres d'exécution
    taille_tuile = 512
    chevauchement = 64
    nombre_iterations = 1200
    
    print("🔭 --- DÉMARRAGE DU PIPELINE ASTROCLEARNET (SR x2) ---")
    
    # 2. Collecte automatique des fichiers FITS présents dans le dossier
    fichiers_cibles = [
        os.path.join(repertoire_donnees, f) 
        for f in os.listdir(repertoire_donnees) 
        if f.endswith('.fits') and not f.startswith(prefixe_sortie)
    ]
    
    # Tri alphabétique pour garantir un ordre constant (la première frame sert d'ancre)
    fichiers_cibles.sort()
    
    if len(fichiers_cibles) < 2:
        print(f"❌ Erreur : Il faut au moins 2 images FITS pour appliquer la Super-Résolution.")
        print(f"Fichiers trouvés : {len(fichiers_cibles)}. Fin du programme.")
        return

    print(f"✨ {len(fichiers_cibles)} expositions brutes détectées pour le traitement.")
    for i, path in enumerate(fichiers_cibles):
        print(f"  [{i}] -> {os.path.basename(path)}")
        
    print(f"⚙️ Configuration : Tuiles={taille_tuile}px, Overlap={chevauchement}px, Itérations={nombre_iterations}")
    
    # 3. Exécution sécurisée de la chaîne globale par tuiles
    try:
        run_astro_clearnet_pipeline(
            fits_paths=fichiers_cibles, 
            output_prefix=prefixe_sortie, 
            tile_size=taille_tuile, 
            overlap=chevauchement,
            iterations=nombre_iterations
        )
    except Exception as e:
        print(f"❌ Une erreur critique est survenue durant l'optimisation : {str(e)}")
        raise e


if __name__ == "__main__":
    main()
