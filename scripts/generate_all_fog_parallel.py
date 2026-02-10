"""
Parallel fog generator — produces light, medium, and heavy fog images
simultaneously using multiprocessing. Each source image is loaded once,
depth filled once, then 2 variants × 3 severities = 6 foggy images produced.

Skips images that already exist (safe to restart).
"""

import cv2
import numpy as np
import os
import random
import json
from multiprocessing import Pool, cpu_count
import time

# ==============================
# PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAR_IMG_DIR = os.path.join(BASE_DIR, "images_clear")
DEPTH_DIR     = os.path.join(BASE_DIR, "depth_maps")

OUTPUT_DIRS = {
    "light":  os.path.join(BASE_DIR, "images_foggy", "light"),
    "medium": os.path.join(BASE_DIR, "images_foggy", "medium"),
    "heavy":  os.path.join(BASE_DIR, "images_foggy", "heavy"),
}
META_DIRS = {
    "light":  os.path.join(BASE_DIR, "metadata", "light"),
    "medium": os.path.join(BASE_DIR, "metadata", "medium"),
    "heavy":  os.path.join(BASE_DIR, "metadata", "heavy"),
}

for d in list(OUTPUT_DIRS.values()) + list(META_DIRS.values()):
    os.makedirs(d, exist_ok=True)

# ==============================
# FOG PARAMETERS PER SEVERITY
# ==============================

SEVERITY_PARAMS = {
    "light": {
        "beta": (0.02, 0.05),
        "A_R":  (0.75, 0.85),
        "A_G":  (0.78, 0.88),
        "A_B":  (0.80, 0.90),
    },
    "medium": {
        "beta": (0.06, 0.10),
        "A_R":  (0.80, 0.90),
        "A_G":  (0.82, 0.92),
        "A_B":  (0.84, 0.94),
    },
    "heavy": {
        "beta": (0.12, 0.25),
        "A_R":  (0.85, 0.98),
        "A_G":  (0.87, 0.98),
        "A_B":  (0.88, 0.99),
    },
}

DEPTH_SCALE = 25.0
NUM_VARIANTS = 2
SEED = 42

# ==============================
# DEPTH FILLING (same as individual generators)
# ==============================

def fill_depth_map(depth):
    H, W = depth.shape
    valid = (depth > 0.01) & (depth < 0.99)

    depth_uint8 = (depth * 255).astype(np.uint8)
    invalid_mask = (~valid).astype(np.uint8) * 255

    filled = cv2.inpaint(
        depth_uint8, invalid_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA
    ).astype(np.float32) / 255.0

    # Windshield gradient falloff
    windshield_cut = int(0.88 * H)
    last_valid = filled[windshield_cut - 5:windshield_cut, :].mean(axis=0)
    last_valid = cv2.GaussianBlur(
        last_valid.reshape(1, -1), (1, 201), 0
    ).flatten()
    median_val = np.median(last_valid)
    last_valid = np.clip(last_valid, 0, median_val * 2)

    for row in range(windshield_cut, H):
        alpha = (row - windshield_cut) / (H - windshield_cut)
        filled[row, :] = last_valid * (1 - alpha)

    # Bilateral filter (edge-preserving)
    filled_u8 = (filled * 255).astype(np.uint8)
    filled = cv2.bilateralFilter(
        filled_u8, d=9, sigmaColor=50, sigmaSpace=50
    ).astype(np.float32) / 255.0

    # Extra windshield blur
    ws = filled[windshield_cut:, :]
    ws = cv2.GaussianBlur(ws, (31, 31), 0)
    filled[windshield_cut:, :] = ws

    return filled


def process_one_image(args):
    """Process a single source image → 2 variants × 3 severities = 6 outputs."""
    idx, img_file = args

    base = img_file.replace(".png", "")
    img_path = os.path.join(CLEAR_IMG_DIR, img_file)
    depth_path = os.path.join(DEPTH_DIR, base + "_depth.npy")

    if not os.path.exists(depth_path):
        return 0

    # Deterministic per-image seed so results are reproducible
    rng = random.Random(SEED + idx)

    # Load once
    img = cv2.imread(img_path).astype(np.float32) / 255.0
    depth = np.load(depth_path)
    depth_filled = fill_depth_map(depth)

    count = 0

    for severity, params in SEVERITY_PARAMS.items():
        for var in range(NUM_VARIANTS):
            # Build output filename
            out_name = f"{base}_{severity}fog_v{var}_beta"
            # Generate random params from this image's RNG
            beta = rng.uniform(*params["beta"])
            A = np.array([
                rng.uniform(*params["A_R"]),
                rng.uniform(*params["A_G"]),
                rng.uniform(*params["A_B"]),
            ])

            out_name = f"{base}_{severity}fog_v{var}_beta{beta:.3f}.jpg"
            out_path = os.path.join(OUTPUT_DIRS[severity], out_name)
            meta_path = os.path.join(META_DIRS[severity], out_name.replace(".jpg", ".json"))

            # Skip if already exists (resume-safe)
            if os.path.exists(out_path) and os.path.exists(meta_path):
                count += 1
                continue

            # Transmission & fog
            t = np.exp(-beta * depth_filled * DEPTH_SCALE)
            t = np.clip(t, 0, 1)

            foggy = img * t[:, :, None] + A * (1 - t[:, :, None])
            foggy = np.clip(foggy, 0, 1)

            out_img = (foggy * 255).astype(np.uint8)
            cv2.imwrite(out_path, out_img)

            meta = {
                "source_image": img_file,
                "fog_type": "fog",
                "severity": severity,
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
                "random_seed": SEED + idx
            }

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            count += 1

    return count


def main():
    # Gather all source images
    all_images = sorted([
        f for f in os.listdir(CLEAR_IMG_DIR) if f.endswith(".png")
    ])
    print(f"Source images: {len(all_images)}")
    print(f"Target per severity: {len(all_images) * NUM_VARIANTS} (2 variants each)")
    print(f"Total target: {len(all_images) * NUM_VARIANTS * 3}")

    # Check existing counts
    for sev in ["light", "medium", "heavy"]:
        existing = len([f for f in os.listdir(OUTPUT_DIRS[sev]) if f.endswith(".jpg")])
        print(f"  {sev}: {existing} already exist")

    # Use all available cores minus 1
    workers = max(1, cpu_count() - 1)
    print(f"\nUsing {workers} parallel workers...")
    print(f"Started at {time.strftime('%H:%M:%S')}\n")

    start = time.time()

    # Create indexed arg list
    work = list(enumerate(all_images))

    with Pool(processes=workers) as pool:
        results = []
        for i, count in enumerate(pool.imap_unordered(process_one_image, work, chunksize=4)):
            results.append(count)
            done = len(results)
            if done % 100 == 0 or done == len(all_images):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(all_images) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(all_images)}] images processed "
                      f"({done * NUM_VARIANTS * 3} fog images) "
                      f"| {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    elapsed = time.time() - start

    # Final counts
    print(f"\n{'='*50}")
    print(f"Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    for sev in ["light", "medium", "heavy"]:
        n = len([f for f in os.listdir(OUTPUT_DIRS[sev]) if f.endswith(".jpg")])
        print(f"  {sev}: {n} images")
    total = sum(
        len([f for f in os.listdir(OUTPUT_DIRS[s]) if f.endswith(".jpg")])
        for s in ["light", "medium", "heavy"]
    )
    print(f"  TOTAL: {total} foggy images")
    print(f"✅ All fog generation completed!")


if __name__ == "__main__":
    main()
