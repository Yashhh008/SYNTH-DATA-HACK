"""
Step 2: Fast fog generation using pre-filled depth maps.
No inpainting needed — just load cached depth + apply fog math.
Generates all 3 severities × 2 variants = 6 images per source.
Uses multiprocessing for speed.
"""

import cv2
import numpy as np
import os
import random
import json
from multiprocessing import Pool, cpu_count
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAR_IMG_DIR = os.path.join(BASE_DIR, "images_clear")
FILLED_DIR    = os.path.join(BASE_DIR, "depth_maps_filled")

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


def process_one_image(args):
    idx, img_file = args

    base = img_file.replace(".png", "")
    img_path = os.path.join(CLEAR_IMG_DIR, img_file)
    filled_path = os.path.join(FILLED_DIR, base + "_filled.npy")

    if not os.path.exists(filled_path):
        return 0

    rng = random.Random(SEED + idx)

    img = cv2.imread(img_path).astype(np.float32) / 255.0
    depth_filled = np.load(filled_path)

    count = 0

    for severity, params in SEVERITY_PARAMS.items():
        for var in range(NUM_VARIANTS):
            beta = rng.uniform(*params["beta"])
            A = np.array([
                rng.uniform(*params["A_R"]),
                rng.uniform(*params["A_G"]),
                rng.uniform(*params["A_B"]),
            ])

            out_name = f"{base}_{severity}fog_v{var}_beta{beta:.3f}.jpg"
            out_path = os.path.join(OUTPUT_DIRS[severity], out_name)
            meta_path = os.path.join(META_DIRS[severity], out_name.replace(".jpg", ".json"))

            # Skip existing (resume-safe)
            if os.path.exists(out_path) and os.path.exists(meta_path):
                count += 1
                continue

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
    all_images = sorted([f for f in os.listdir(CLEAR_IMG_DIR) if f.endswith(".png")])
    print(f"Source images: {len(all_images)}")
    print(f"Target: {len(all_images) * NUM_VARIANTS} per severity, {len(all_images) * NUM_VARIANTS * 3} total")

    for sev in ["light", "medium", "heavy"]:
        existing = len([f for f in os.listdir(OUTPUT_DIRS[sev]) if f.endswith(".jpg")])
        print(f"  {sev}: {existing} already exist")

    workers = max(1, cpu_count() - 1)
    print(f"\nUsing {workers} parallel workers (fog math only, no inpainting)")
    print(f"Started at {time.strftime('%H:%M:%S')}\n")

    start = time.time()
    work = list(enumerate(all_images))

    with Pool(processes=workers) as pool:
        results = []
        for i, count in enumerate(pool.imap_unordered(process_one_image, work, chunksize=8)):
            results.append(count)
            done = len(results)
            if done % 200 == 0 or done == len(all_images):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(all_images) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(all_images)}] {elapsed:.0f}s | ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    total = 0
    for sev in ["light", "medium", "heavy"]:
        n = len([f for f in os.listdir(OUTPUT_DIRS[sev]) if f.endswith(".jpg")])
        print(f"  {sev}: {n} images")
        total += n
    print(f"  TOTAL: {total} foggy images")
    print(f"✅ All fog generation completed!")


if __name__ == "__main__":
    main()
