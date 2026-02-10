import cv2
import numpy as np
import os
import random
import json

# ==============================
# PATHS
# ==============================

CLEAR_IMG_DIR = "images_clear"
DEPTH_DIR = "depth_maps"
OUTPUT_DIR = "images_foggy/light"
META_DIR = "metadata/light"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# ==============================
# LIGHT FOG PARAMETERS
# ==============================

BETA_MIN = 0.02
BETA_MAX = 0.05

# Atmospheric light (RGB)
A_R_MIN, A_R_MAX = 0.75, 0.85
A_G_MIN, A_G_MAX = 0.78, 0.88
A_B_MIN, A_B_MAX = 0.80, 0.90

DEPTH_SCALE = 25.0
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def fill_depth_map(depth):
    """
    Fill invalid stereo depth regions via Telea inpainting, then smooth
    using an edge-preserving bilateral filter (not global Gaussian) to
    retain object-level depth contrast.

    Windshield/hood region: gradient falloff from last valid row
    (not a hard-coded constant).

    NOTE: Inpainted regions represent estimated, not measured, geometry.
    This prioritizes visual realism over strict geometric accuracy.
    """
    H, W = depth.shape

    # Identify valid pixels (reliable stereo depth)
    valid = (depth > 0.01) & (depth < 0.99)

    # Step 1: Inpaint invalid regions from valid neighbors
    depth_uint8 = (depth * 255).astype(np.uint8)
    invalid_mask = (~valid).astype(np.uint8) * 255

    filled = cv2.inpaint(
        depth_uint8, invalid_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA
    ).astype(np.float32) / 255.0

    # Step 2: Windshield region — smooth gradient falloff
    windshield_cut = int(0.88 * H)
    # Average depth from rows just above cut, then heavily smooth
    # to remove per-column spikes from noisy stereo edges
    last_valid = filled[windshield_cut - 5:windshield_cut, :].mean(axis=0)
    last_valid = cv2.GaussianBlur(
        last_valid.reshape(1, -1), (1, 201), 0
    ).flatten()
    # Clamp outlier spikes to the median (robust to noise)
    median_val = np.median(last_valid)
    last_valid = np.clip(last_valid, 0, median_val * 2)

    for row in range(windshield_cut, H):
        alpha = (row - windshield_cut) / (H - windshield_cut)
        filled[row, :] = last_valid * (1 - alpha)  # fade toward 0 (near)

    # Step 3: Edge-preserving bilateral filter on the scene region
    filled_u8 = (filled * 255).astype(np.uint8)
    filled = cv2.bilateralFilter(
        filled_u8, d=9, sigmaColor=50, sigmaSpace=50
    ).astype(np.float32) / 255.0

    # Step 4: Extra Gaussian blur on windshield zone only
    # ensures perfectly uniform fog on hood/dash area
    ws = filled[windshield_cut:, :]
    ws = cv2.GaussianBlur(ws, (31, 31), 0)
    filled[windshield_cut:, :] = ws

    return filled


# ==============================
# MAIN LOOP — 2 variants per image → 5950 total
# ==============================

NUM_VARIANTS = 2
_total = 0

for img_file in sorted(os.listdir(CLEAR_IMG_DIR)):
    if not img_file.endswith(".png"):
        continue

    base = img_file.replace(".png", "")
    img_path = os.path.join(CLEAR_IMG_DIR, img_file)
    depth_path = os.path.join(DEPTH_DIR, base + "_depth.npy")

    if not os.path.exists(depth_path):
        continue

    # Load data once per image
    img = cv2.imread(img_path).astype(np.float32) / 255.0
    depth = np.load(depth_path)

    H, W = depth.shape

    # ==============================
    # FILL & SMOOTH DEPTH (NO HOLES)
    # ==============================

    depth_filled = fill_depth_map(depth)

    for var in range(NUM_VARIANTS):
        # ==============================
        # FOG PARAMETERS (unique per variant)
        # ==============================

        beta = random.uniform(BETA_MIN, BETA_MAX)
        A = np.array([
            random.uniform(A_R_MIN, A_R_MAX),
            random.uniform(A_G_MIN, A_G_MAX),
            random.uniform(A_B_MIN, A_B_MAX)
        ])

        # Transmission map
        t = np.exp(-beta * depth_filled * DEPTH_SCALE)
        t = np.clip(t, 0, 1)

        # ==============================
        # APPLY FOG
        # ==============================

        foggy = img * t[:, :, None] + A * (1 - t[:, :, None])
        foggy = np.clip(foggy, 0, 1)

        # Save image
        out_img = (foggy * 255).astype(np.uint8)
        out_name = f"{base}_lightfog_v{var}_beta{beta:.3f}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), out_img)

        # ==============================
        # METADATA
        # ==============================

        meta = {
            "source_image": img_file,
            "fog_type": "fog",
            "severity": "light",
            "variant": var,
            "beta": round(beta, 4),
            "atmospheric_light_rgb": [round(A[0], 4), round(A[1], 4), round(A[2], 4)],
            "depth_scale": DEPTH_SCALE,
            "depth_source": "Cityscapes stereo disparity (inverse + log-scaled)",
            "depth_invalid_handling": {
                "method": "Telea inpainting + bilateral filter",
                "inpaint_radius": 10,
                "bilateral_d": 9,
                "bilateral_sigmaColor": 50,
                "bilateral_sigmaSpace": 50,
                "windshield_handling": "gradient falloff from last valid row",
                "note": "Inpainted regions are estimated depth for visual continuity, not measured geometry"
            },
            "random_seed": SEED
        }

        with open(
            os.path.join(META_DIR, out_name.replace(".jpg", ".json")),
            "w"
        ) as f:
            json.dump(meta, f, indent=2)

        _total += 1

    if _total % 500 == 0:
        print(f"  ... {_total} images generated")

print(f"✅ Light fog generation completed: {_total} images.")
