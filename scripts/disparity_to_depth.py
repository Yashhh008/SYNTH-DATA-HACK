import cv2
import numpy as np
import os

DISPARITY_ROOT = "raw_cityscapes/disparity/train"
OUTPUT_DEPTH = "depth_maps"

os.makedirs(OUTPUT_DEPTH, exist_ok=True)

for city in os.listdir(DISPARITY_ROOT):
    city_path = os.path.join(DISPARITY_ROOT, city)

    for file in os.listdir(city_path):
        if not file.endswith("_disparity.png"):
            continue

        disp_raw = cv2.imread(
            os.path.join(city_path, file),
            cv2.IMREAD_UNCHANGED
        ).astype(np.float32)

        # 1️⃣ Convert from fixed-point disparity
        disp = disp_raw / 256.0

        # 2️⃣ Mask invalid disparity
        valid = disp > 0

        # 3️⃣ Convert to depth-like quantity
        depth = np.zeros_like(disp)
        depth[valid] = 1.0 / disp[valid]

        # 4️⃣ Clip extreme depth values (CRITICAL)
        depth[valid] = np.clip(
            depth[valid],
            np.percentile(depth[valid], 5),
            np.percentile(depth[valid], 95)
        )

        # 5️⃣ Log scaling (CRITICAL)
        depth[valid] = np.log(depth[valid] + 1.0)

        # 6️⃣ Normalize to [0, 1]
        dmin = depth[valid].min()
        dmax = depth[valid].max()
        depth_norm = np.zeros_like(depth)
        depth_norm[valid] = (depth[valid] - dmin) / (dmax - dmin)

        # Invalid → far background
        depth_norm[~valid] = 1.0

        base = file.replace("_disparity.png", "")
        np.save(os.path.join(OUTPUT_DEPTH, base + "_depth.npy"), depth_norm)

print("✅ Robust depth maps generated (log-scaled).")
