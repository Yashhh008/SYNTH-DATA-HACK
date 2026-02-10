"""
Fast parallel fog generator — all 3 severities at once.
1. Pre-fills & caches depth maps (skip already cached ones)
2. Generates fog images using cached depths (pure math, very fast)
3. Uses multiprocessing Pool for speed
4. Skips already-generated images
"""

import cv2
import numpy as np
import os
import random
import json
from multiprocessing import Pool, cpu_count

# ==============================
# PATHS
# ==============================
BASE = "C:/Users/yashw/cityscapes_project"
CLEAR_IMG_DIR = os.path.join(BASE, "images_clear")
DEPTH_DIR = os.path.join(BASE, "depth_maps")
FILLED_DIR = os.path.join(BASE, "depth_maps_filled")

SEVERITY_CONFIG = {
    "light": {
        "out_dir": os.path.join(BASE, "images_foggy/light"),
        "meta_dir": os.path.join(BASE, "metadata/light"),
        "beta": (0.02, 0.05),
        "A_R": (0.75, 0.85), "A_G": (0.78, 0.88), "A_B": (0.80, 0.90),
    },
    "medium": {
        "out_dir": os.path.join(BASE, "images_foggy/medium"),
        "meta_dir": os.path.join(BASE, "metadata/medium"),
        "beta": (0.06, 0.10),
        "A_R": (0.80, 0.90), "A_G": (0.82, 0.92), "A_B": (0.84, 0.94),
    },
    "heavy": {
        "out_dir": os.path.join(BASE, "images_foggy/heavy"),
        "meta_dir": os.path.join(BASE, "metadata/heavy"),
        "beta": (0.12, 0.25),
        "A_R": (0.85, 0.98), "A_G": (0.87, 0.98), "A_B": (0.88, 0.99),
    },
}

DEPTH_SCALE = 25.0
NUM_VARIANTS = 2
SEED = 42

for cfg in SEVERITY_CONFIG.values():
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["meta_dir"], exist_ok=True)
os.makedirs(FILLED_DIR, exist_ok=True)


# ==============================
# DEPTH FILLING (same as before)
# ==============================
def fill_depth_map(depth):
    H, W = depth.shape
    valid = (depth > 0.01) & (depth < 0.99)
    depth_uint8 = (depth * 255).astype(np.uint8)
    invalid_mask = (~valid).astype(np.uint8) * 255
    filled = cv2.inpaint(
        depth_uint8, invalid_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA
    ).astype(np.float32) / 255.0

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

    filled_u8 = (filled * 255).astype(np.uint8)
    filled = cv2.bilateralFilter(
        filled_u8, d=9, sigmaColor=50, sigmaSpace=50
    ).astype(np.float32) / 255.0

    ws = filled[windshield_cut:, :]
    ws = cv2.GaussianBlur(ws, (31, 31), 0)
    filled[windshield_cut:, :] = ws
    return filled


def fill_one_depth(base):
    """Fill & cache one depth map (skip if already exists)."""
    out_path = os.path.join(FILLED_DIR, base + "_depth_filled.npy")
    if os.path.exists(out_path):
        return  # already cached
    src_path = os.path.join(DEPTH_DIR, base + "_depth.npy")
    if not os.path.exists(src_path):
        return
    depth = np.load(src_path)
    filled = fill_depth_map(depth)
    np.save(out_path, filled)


def generate_fog_for_image(args):
    """Generate all 3 severities × 2 variants for one image."""
    base, seed_offset = args
    rng = random.Random(SEED + seed_offset)

    img_path = os.path.join(CLEAR_IMG_DIR, base + ".png")
    filled_path = os.path.join(FILLED_DIR, base + "_depth_filled.npy")

    if not os.path.exists(img_path) or not os.path.exists(filled_path):
        return 0

    img = cv2.imread(img_path).astype(np.float32) / 255.0
    depth_filled = np.load(filled_path)
    count = 0

    for sev_name, cfg in SEVERITY_CONFIG.items():
        for var in range(NUM_VARIANTS):
            beta = rng.uniform(*cfg["beta"])
            A = np.array([
                rng.uniform(*cfg["A_R"]),
                rng.uniform(*cfg["A_G"]),
                rng.uniform(*cfg["A_B"]),
            ])

            out_name = f"{base}_{sev_name}fog_v{var}_beta{beta:.3f}.jpg"
            out_path = os.path.join(cfg["out_dir"], out_name)

            # Skip if already exists
            if os.path.exists(out_path):
                count += 1
                continue

            t = np.exp(-beta * depth_filled * DEPTH_SCALE)
            t = np.clip(t, 0, 1)
            foggy = img * t[:, :, None] + A * (1 - t[:, :, None])
            foggy = np.clip(foggy, 0, 1)

            out_img = (foggy * 255).astype(np.uint8)
            cv2.imwrite(out_path, out_img)

            meta = {
                "source_image": base + ".png",
                "fog_type": "fog",
                "severity": sev_name,
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
            meta_path = os.path.join(cfg["meta_dir"], out_name.replace(".jpg", ".json"))
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            count += 1
    return count


if __name__ == "__main__":
    import time

    # Get all image bases
    all_bases = sorted([
        f.replace(".png", "") for f in os.listdir(CLEAR_IMG_DIR) if f.endswith(".png")
    ])
    print(f"Total source images: {len(all_bases)}")

    # ========== STEP 1: Fill depth maps (slow, uses inpainting) ==========
    already_filled = set(
        f.replace("_depth_filled.npy", "") for f in os.listdir(FILLED_DIR) if f.endswith(".npy")
    )
    to_fill = [b for b in all_bases if b not in already_filled]
    print(f"Depth maps: {len(already_filled)} cached, {len(to_fill)} to fill...")

    if to_fill:
        t0 = time.time()
        workers = max(1, cpu_count() - 2)
        print(f"Filling depths with {workers} workers...")
        with Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(fill_one_depth, to_fill), 1):
                if i % 100 == 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed * 60
                    remaining = (len(to_fill) - i) / (rate / 60)
                    print(f"  Filled {i}/{len(to_fill)} depths ({rate:.0f}/min, ~{remaining:.0f}s left)")
        print(f"  Depth filling done in {time.time()-t0:.0f}s")

    # ========== STEP 2: Generate fog images (fast, just math) ==========
    print(f"\nGenerating fog images (3 severities × 2 variants = 6 per image)...")
    t0 = time.time()
    args_list = [(b, i) for i, b in enumerate(all_bases)]
    workers = max(1, cpu_count() - 2)
    print(f"Using {workers} workers...")

    total = 0
    with Pool(workers) as pool:
        for i, cnt in enumerate(pool.imap_unordered(generate_fog_for_image, args_list), 1):
            total += cnt
            if i % 200 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed * 60
                remaining = (len(all_bases) - i) / (rate / 60)
                print(f"  Processed {i}/{len(all_bases)} images, {total} fog outputs ({rate:.0f}/min, ~{remaining:.0f}s left)")

    elapsed = time.time() - t0
    print(f"\n✅ ALL DONE in {elapsed:.0f}s")
    print(f"   Total fog images generated/verified: {total}")

    # Final counts
    for sev in ["light", "medium", "heavy"]:
        d = os.path.join(BASE, f"images_foggy/{sev}")
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"   {sev}: {n} images")
