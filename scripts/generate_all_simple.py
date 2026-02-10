"""
Simple single-process fog generator. No multiprocessing (avoids deadlocks).
- Fills depth maps that aren't cached yet
- Generates all 3 severities × 2 variants per image
- Skips already-generated fog images
- Prints progress with flush for real-time monitoring
"""
import cv2
import numpy as np
import os
import random
import json
import sys
import time

BASE = "C:/Users/yashw/cityscapes_project"
CLEAR_DIR = os.path.join(BASE, "images_clear")
DEPTH_DIR = os.path.join(BASE, "depth_maps")
FILLED_DIR = os.path.join(BASE, "depth_maps_filled")

SEVERITIES = {
    "light": {
        "out": os.path.join(BASE, "images_foggy/light"),
        "meta": os.path.join(BASE, "metadata/light"),
        "beta": (0.02, 0.05),
        "A_R": (0.75, 0.85), "A_G": (0.78, 0.88), "A_B": (0.80, 0.90),
    },
    "medium": {
        "out": os.path.join(BASE, "images_foggy/medium"),
        "meta": os.path.join(BASE, "metadata/medium"),
        "beta": (0.06, 0.10),
        "A_R": (0.80, 0.90), "A_G": (0.82, 0.92), "A_B": (0.84, 0.94),
    },
    "heavy": {
        "out": os.path.join(BASE, "images_foggy/heavy"),
        "meta": os.path.join(BASE, "metadata/heavy"),
        "beta": (0.12, 0.25),
        "A_R": (0.85, 0.98), "A_G": (0.87, 0.98), "A_B": (0.88, 0.99),
    },
}

DEPTH_SCALE = 25.0
NUM_VARIANTS = 2
SEED = 42

for cfg in SEVERITIES.values():
    os.makedirs(cfg["out"], exist_ok=True)
    os.makedirs(cfg["meta"], exist_ok=True)
os.makedirs(FILLED_DIR, exist_ok=True)


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
    last_valid = cv2.GaussianBlur(last_valid.reshape(1, -1), (1, 201), 0).flatten()
    median_val = np.median(last_valid)
    last_valid = np.clip(last_valid, 0, median_val * 2)
    for row in range(windshield_cut, H):
        alpha = (row - windshield_cut) / (H - windshield_cut)
        filled[row, :] = last_valid * (1 - alpha)

    filled_u8 = (filled * 255).astype(np.uint8)
    filled = cv2.bilateralFilter(filled_u8, d=9, sigmaColor=50, sigmaSpace=50).astype(np.float32) / 255.0

    ws = filled[windshield_cut:, :]
    ws = cv2.GaussianBlur(ws, (31, 31), 0)
    filled[windshield_cut:, :] = ws
    return filled


def log(msg):
    print(msg, flush=True)


if __name__ == "__main__":
    t0 = time.time()

    # All source bases
    all_bases = sorted(f.replace(".png", "") for f in os.listdir(CLEAR_DIR) if f.endswith(".png"))
    log(f"Source images: {len(all_bases)}")

    # Already filled depths
    filled_set = set()
    for f in os.listdir(FILLED_DIR):
        if f.endswith("_depth_filled.npy"):
            filled_set.add(f.replace("_depth_filled.npy", ""))

    log(f"Filled depths cached: {len(filled_set)}, remaining: {len(all_bases) - len(filled_set)}")

    total_fog = 0
    skipped_fog = 0

    for idx, base in enumerate(all_bases):
        # -- Step A: ensure filled depth exists --
        filled_path = os.path.join(FILLED_DIR, base + "_depth_filled.npy")
        if base not in filled_set:
            raw_path = os.path.join(DEPTH_DIR, base + "_depth.npy")
            if not os.path.exists(raw_path):
                continue
            depth = np.load(raw_path)
            filled = fill_depth_map(depth)
            np.save(filled_path, filled)
            filled_set.add(base)

        # -- Step B: generate fog for all severities --
        img = None  # lazy load
        depth_filled = None
        rng = random.Random(SEED + idx)

        for sev_name, cfg in SEVERITIES.items():
            for var in range(NUM_VARIANTS):
                beta = rng.uniform(*cfg["beta"])
                A = np.array([
                    rng.uniform(*cfg["A_R"]),
                    rng.uniform(*cfg["A_G"]),
                    rng.uniform(*cfg["A_B"]),
                ])
                out_name = f"{base}_{sev_name}fog_v{var}_beta{beta:.3f}.jpg"
                out_path = os.path.join(cfg["out"], out_name)

                if os.path.exists(out_path):
                    skipped_fog += 1
                    total_fog += 1
                    continue

                # Lazy load image + depth on first need
                if img is None:
                    img = cv2.imread(os.path.join(CLEAR_DIR, base + ".png")).astype(np.float32) / 255.0
                    depth_filled = np.load(filled_path)

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
                meta_path = os.path.join(cfg["meta"], out_name.replace(".jpg", ".json"))
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                total_fog += 1

        # Progress every 50 images
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed * 60
            eta = (len(all_bases) - idx - 1) / (rate / 60)
            log(f"[{idx+1}/{len(all_bases)}] fog={total_fog} skipped={skipped_fog} rate={rate:.0f}img/min ETA={eta:.0f}s")

    elapsed = time.time() - t0
    log(f"\n=== DONE in {elapsed:.0f}s ===")
    log(f"Total fog images: {total_fog} (skipped existing: {skipped_fog})")
    for sev in ["light", "medium", "heavy"]:
        d = os.path.join(BASE, f"images_foggy/{sev}")
        log(f"  {sev}: {len(os.listdir(d))}")
