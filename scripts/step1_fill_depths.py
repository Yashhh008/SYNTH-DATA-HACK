"""
Step 1: Pre-compute filled depth maps and cache as .npy files.
This is the slow step (inpainting). Run once, then fog generation is fast.
"""

import cv2
import numpy as np
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPTH_DIR = os.path.join(BASE_DIR, "depth_maps")
FILLED_DIR = os.path.join(BASE_DIR, "depth_maps_filled")

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


def main():
    depth_files = sorted([f for f in os.listdir(DEPTH_DIR) if f.endswith("_depth.npy")])
    print(f"Total depth maps: {len(depth_files)}")

    # Check how many already done
    already = set(os.listdir(FILLED_DIR))
    todo = [f for f in depth_files if f.replace("_depth.npy", "_filled.npy") not in already]
    print(f"Already filled: {len(depth_files) - len(todo)}")
    print(f"Remaining: {len(todo)}")

    start = time.time()

    for i, dfile in enumerate(todo):
        base = dfile.replace("_depth.npy", "")
        depth = np.load(os.path.join(DEPTH_DIR, dfile))
        filled = fill_depth_map(depth)
        np.save(os.path.join(FILLED_DIR, base + "_filled.npy"), filled)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate
            print(f"  [{i+1}/{len(todo)}] {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"\n✅ Depth filling complete: {len(depth_files)} maps in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
