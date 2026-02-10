import numpy as np
import cv2
import os

DEPTH_DIR = "depth_maps"
DEBUG_DIR = "debug"

os.makedirs(DEBUG_DIR, exist_ok=True)

# pick ONE depth file
depth_files = [f for f in os.listdir(DEPTH_DIR) if f.endswith("_depth.npy")]
depth_file = depth_files[0]

depth = np.load(os.path.join(DEPTH_DIR, depth_file))

# --------- KEY FIX HERE ---------
# Use percentile-based normalization for visualization
low = np.percentile(depth, 5)
high = np.percentile(depth, 95)

depth_vis = np.clip(depth, low, high)
depth_vis = (depth_vis - low) / (high - low)
depth_vis = (depth_vis * 255).astype("uint8")
# --------------------------------

out_path = os.path.join(DEBUG_DIR, depth_file.replace(".npy", "_depth_debug.png"))
cv2.imwrite(out_path, depth_vis)

print(f"✅ Depth visualization saved at: {out_path}")
