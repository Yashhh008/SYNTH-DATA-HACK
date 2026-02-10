"""Draw YOLO bounding boxes on 5 clear images and save to debug/bbox_samples/."""
import cv2
import numpy as np
import os

BASE = "C:/Users/yashw/cityscapes_project"
CLEAR_DIR = os.path.join(BASE, "images_clear")
ANNOT_DIR = os.path.join(BASE, "annotations")
OUT_DIR = os.path.join(BASE, "debug", "bbox_samples")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = {0: "person", 1: "car", 2: "bicycle", 3: "motorcycle"}
COLORS = {
    0: (0, 255, 0),     # person - green
    1: (255, 0, 0),     # car - blue
    2: (0, 255, 255),   # bicycle - yellow
    3: (255, 0, 255),   # motorcycle - magenta
}

# 5 images from different cities with many objects
IMAGES = [
    "aachen_000004_000019",     # 25 objects
    "aachen_000010_000019",     # 27 objects
    "bremen_000130_000019",     # from bremen
    "dusseldorf_000077_000019", # from dusseldorf
    "hamburg_000000_061048",    # from hamburg
]

for base_name in IMAGES:
    img_path = os.path.join(CLEAR_DIR, base_name + ".png")
    annot_path = os.path.join(ANNOT_DIR, base_name + ".txt")

    if not os.path.exists(img_path) or not os.path.exists(annot_path):
        print(f"Skipping {base_name} — missing file")
        continue

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    with open(annot_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    obj_count = 0
    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Convert YOLO normalized to pixel coords
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)

        color = COLORS.get(cls_id, (255, 255, 255))
        label = CLASSES.get(cls_id, f"cls{cls_id}")

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background + text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 6), (x1 + text_size[0] + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        obj_count += 1

    out_path = os.path.join(OUT_DIR, f"{base_name}_bbox.jpg")
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out_path}  ({obj_count} objects)")

print("\nDone! 5 images with bounding boxes saved to debug/bbox_samples/")
