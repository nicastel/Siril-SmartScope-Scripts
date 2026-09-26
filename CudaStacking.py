import sirilpy as s
s.ensure_installed("cupy-cuda12x", "numpy", "astropy")

import cupy as cp
import numpy as np
from astropy.io import fits

# Kernel CUDA optimisé pour le Bayer CFA (Pattern RGGB)
# Il distribue chaque pixel natif directement dans son canal couleur de destination.
bayer_drizzle_kernel = cp.RawKernel(r'''
extern "C" __global__
void bayer_drizzle_cuda(
    const float* input, float* output, float* weights,
    int H_in, int W_in, int H_out, int W_out,
    float scale, float pixfrac,
    float dx, float dy, float cos_t, float sin_t
) {
    int x_in = blockIdx.x * blockDim.x + threadIdx.x;
    int y_in = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_in >= W_in || y_in >= H_in) return;

    float val = input[y_in * W_in + x_in];
    if (isnan(val)) return;

    // --- IDENTIFICATION DU CANAL COULEUR (Pattern RGGB) ---
    // Les indices pairs/impairs déterminent la couleur du pixel natif
    int c = 0; 
    if (y_in % 2 == 0) {
        if (x_in % 2 == 0)      c = 0; // Rouge (R)
        else                    c = 1; // Vert_R (G)
    } else {
        if (x_in % 2 == 0)      c = 1; // Vert_B (G)
        else                    c = 2; // Bleu (B)
    }

    // Transformations géométriques (Drizzle standard)
    float xc_in = x_in - W_in / 2.0f;
    float yc_in = y_in - H_in / 2.0f;
    float xc_out = (xc_in * cos_t - yc_in * sin_t + dx) * scale;
    float yc_out = (xc_in * sin_t + yc_in * cos_t + dy) * scale;

    float target_x = xc_out + W_out / 2.0f;
    float target_y = yc_out + H_out / 2.0f;

    float r = (pixfrac * scale) / 2.0f;
    float x1 = target_x - r;
    float x2 = target_x + r;
    float y1 = target_y - r;
    float y2 = target_y + r;

    int x_start = max(0, (int)floorf(x1));
    int x_end   = min(W_out - 1, (int)ceilf(x2));
    int y_start = max(0, (int)floorf(y1));
    int y_end   = min(H_out - 1, (int)ceilf(y2));

    for (int y = y_start; y <= y_end; ++y) {
        float y_overlap = min((float)y + 0.5f, y2) - max((float)y - 0.5f, y1);
        if (y_overlap <= 0.0f) continue;

        for (int x = x_start; x <= x_end; ++x) {
            float x_overlap = min((float)x + 0.5f, x2) - max((float)x - 0.5f, x1);
            if (x_overlap <= 0.0f) continue;

            float w = x_overlap * y_overlap;
            
            // Calcul de l'index 3D aplati : [Canal, Y, X] -> (c * H_out * W_out) + (y * W_out) + x
            int out_idx = (c * H_out * W_out) + (y * W_out) + x;

            atomicAdd(&output[out_idx], val * w);
            atomicAdd(&weights[out_idx], w);
        }
    }
}
''', 'bayer_drizzle_cuda')

def gpu_bayer_drizzle_stack(fits_paths, scale=2.0, pixfrac=0.6):
    """
    Réalise un Bayer Drizzle Stacking matériel (RGGB) directement depuis les fichiers bruts FITS.
    """
    if not fits_paths:
        raise ValueError("La liste des fichiers FITS est vide.")

    # 1. Dimensions de l'image CFA d'origine
    with fits.open(fits_paths[0]) as hdul:
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        print(f"Lecture de l'image CFA d'origine : {fits_paths[0]}")
        print(hdu.data.shape)
        C, H_in, W_in = hdu.data.shape

    H_out, W_out = int(H_in * scale), int(W_in * scale)

    # 2. Allocation VRAM 3D (3 canaux de couleur : R, G, B)
    gpu_output_accum = cp.zeros((3, H_out, W_out), dtype=cp.float32)
    gpu_weight_accum = cp.zeros((3, H_out, W_out), dtype=cp.float32)

    block_dim = (16, 16)
    grid_dim = (int(np.ceil(W_in / 16)), int(np.ceil(H_in / 16)))

    # 3. Boucle de traitement des fichiers FITS
    for i, path in enumerate(fits_paths):
        with fits.open(path) as hdul:
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            img = hdu.data
            header = hdu.header
            
            # Lecture automatique des descripteurs Siril
            dx = float(header.get('DX', 0.0))
            dy = float(header.get('DY', 0.0))
            angle_deg = float(header.get('ANGLE', 0.0))
            angle_rad = np.radians(angle_deg)
            
            print(f"[{i+1}/{len(fits_paths)}] Traitement CFA : {path} -> DX: {dx:.2f}, DY: {dy:.2f}")

            # Envoi de la matrice brute non débayrisée au GPU
            gpu_img = cp.asarray(img, dtype=cp.float32)

        cos_t = float(np.cos(angle_rad))
        sin_t = float(np.sin(angle_rad))

        # Lancement du calcul parallèle pour redistribuer le pattern CFA
        bayer_drizzle_kernel(
            grid_dim, block_dim,
            (gpu_img, gpu_output_accum, gpu_weight_accum,
             H_in, W_in, H_out, W_out,
             float(scale), float(pixfrac), dx, dy, cos_t, sin_t)
        )

    # 4. Normalisation finale par canal
    print("Normalisation et reconstruction de l'image couleur...")
    final_gpu_stack = cp.where(gpu_weight_accum > 0, gpu_output_accum / gpu_weight_accum, 0.0)
    
    # Récupération au format NumPy standard [3, H, W]
    return final_gpu_stack.get(), gpu_weight_accum.get()

# 4. Exemple d'exécution
if __name__ == "__main__":
    mes_images_raw_siril = [f"r_bkg_pp_lights_{i:05d}.fit.fz" for i in range(1, 54)]
    
    # Pour un drizzle couleur réussi, il est fortement conseillé de monter 'scale' à 2.0.
    image_rvb, carte_poids = gpu_bayer_drizzle_stack(mes_images_raw_siril, scale=2.0, pixfrac=0.6)
    
    # Astropy.io.fits préfère le format (Canaux, Hauteur, Largeur) lors de l'enregistrement en cube 3D
    fits.writeto("image_couleur_drizzle.fits", image_rvb, overwrite=True)
    print("Image finale couleur enregistrée au format FITS RVB cube.")
