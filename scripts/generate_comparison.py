"""Generate 4 comparison grids: Clear | Light | Medium | Heavy (1 row × 4 cols each)."""
import cv2
import numpy as np
import os

BASE = "C:/Users/yashw/cityscapes_project"
OUT_DIR = os.path.join(BASE, "debug")
os.makedirs(OUT_DIR, exist_ok=True)

# 4 scenes from different cities
SCENES = [
    {
        "name": "strasbourg",
        "clear": "strasbourg_000001_009246.png",
        "light": "strasbourg_000001_009246_lightfog_v0_beta0.038.jpg",
        "medium": "strasbourg_000001_009246_mediumfog_v0_beta0.066.jpg",
        "heavy": "strasbourg_000001_009246_heavyfog_v0_beta0.211.jpg",
    },
    {
        "name": "hamburg",
        "clear": "hamburg_000000_061048.png",
        "light": "hamburg_000000_061048_lightfog_v0_beta0.036.jpg",
        "medium": "hamburg_000000_061048_mediumfog_v0_beta0.087.jpg",
        "heavy": "hamburg_000000_061048_heavyfog_v0_beta0.169.jpg",
    },
    {
        "name": "cologne",
        "clear": "cologne_000077_000019.png",
        "light": "cologne_000077_000019_lightfog_v0_beta0.043.jpg",
        "medium": "cologne_000077_000019_mediumfog_v0_beta0.094.jpg",
        "heavy": "cologne_000077_000019_heavyfog_v0_beta0.209.jpg",
    },
    {
        "name": "zurich",
        "clear": "zurich_000061_000019.png",
        "light": "zurich_000061_000019_lightfog_v0_beta0.037.jpg",
        "medium": "zurich_000061_000019_mediumfog_v0_beta0.060.jpg",
        "heavy": "zurich_000061_000019_heavyfog_v0_beta0.150.jpg",
    },
]

THUMB_W = 512
THUMB_H = 256
PAD = 4
LABEL_H = 36
FONT = cv2.FONT_HERSHEY_SIMPLEX

for scene in SCENES:
    imgs = []
    labels = ["Clear", "Light Fog", "Medium Fog", "Heavy Fog"]
    paths = [
        os.path.join(BASE, "images_clear", scene["clear"]),
        os.path.join(BASE, "images_foggy/light", scene["light"]),
        os.path.join(BASE, "images_foggy/medium", scene["medium"]),
        os.path.join(BASE, "images_foggy/heavy", scene["heavy"]),
    ]

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  ERROR: cannot read {p}")
            break
        img = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        imgs.append(img)

    if len(imgs) != 4:
        print(f"Skipping {scene['name']} — missing images")
        continue

    # Build grid: 1 row × 4 cols with labels on top
    grid_w = 4 * THUMB_W + 5 * PAD
    grid_h = LABEL_H + THUMB_H + 2 * PAD
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for i, (img, label) in enumerate(zip(imgs, labels)):
        x = PAD + i * (THUMB_W + PAD)
        # Label
        text_size = cv2.getTextSize(label, FONT, 0.7, 2)[0]
        tx = x + (THUMB_W - text_size[0]) // 2
        cv2.putText(canvas, label, (tx, LABEL_H - 10), FONT, 0.7, (0, 0, 0), 2)
        # Image
        y = LABEL_H + PAD
        canvas[y:y + THUMB_H, x:x + THUMB_W] = img

    out_path = os.path.join(OUT_DIR, f"comparison_{scene['name']}.jpg")
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out_path}")

print("\nDone! 4 comparison grids saved to debug/")
