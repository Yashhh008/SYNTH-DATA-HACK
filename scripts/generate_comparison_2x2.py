"""Generate 10 comparison grids in 2x2 layout: Clear|Light / Medium|Heavy."""
import cv2
import numpy as np
import os

BASE = "C:/Users/yashw/cityscapes_project"
OUT_DIR = os.path.join(BASE, "debug")
os.makedirs(OUT_DIR, exist_ok=True)

SCENES = [
    ("aachen", "aachen_000094_000019", "aachen_000094_000019_lightfog_v0_beta0.032.jpg", "aachen_000094_000019_mediumfog_v0_beta0.100.jpg", "aachen_000094_000019_heavyfog_v0_beta0.226.jpg"),
    ("bochum", "bochum_000000_020673", "bochum_000000_020673_lightfog_v0_beta0.028.jpg", "bochum_000000_020673_mediumfog_v0_beta0.095.jpg", "bochum_000000_020673_heavyfog_v0_beta0.169.jpg"),
    ("bremen", "bremen_000130_000019", "bremen_000130_000019_lightfog_v0_beta0.031.jpg", "bremen_000130_000019_mediumfog_v0_beta0.088.jpg", "bremen_000130_000019_heavyfog_v0_beta0.192.jpg"),
    ("darmstadt", "darmstadt_000059_000019", "darmstadt_000059_000019_lightfog_v0_beta0.037.jpg", "darmstadt_000059_000019_mediumfog_v0_beta0.087.jpg", "darmstadt_000059_000019_heavyfog_v0_beta0.230.jpg"),
    ("dusseldorf", "dusseldorf_000077_000019", "dusseldorf_000077_000019_lightfog_v0_beta0.020.jpg", "dusseldorf_000077_000019_mediumfog_v0_beta0.090.jpg", "dusseldorf_000077_000019_heavyfog_v0_beta0.149.jpg"),
    ("erfurt", "erfurt_000041_000019", "erfurt_000041_000019_lightfog_v0_beta0.047.jpg", "erfurt_000041_000019_mediumfog_v0_beta0.068.jpg", "erfurt_000041_000019_heavyfog_v0_beta0.220.jpg"),
    ("hanover", "hanover_000000_027481", "hanover_000000_027481_lightfog_v0_beta0.044.jpg", "hanover_000000_027481_mediumfog_v0_beta0.064.jpg", "hanover_000000_027481_heavyfog_v0_beta0.187.jpg"),
    ("jena", "jena_000037_000019", "jena_000037_000019_lightfog_v0_beta0.028.jpg", "jena_000037_000019_mediumfog_v0_beta0.073.jpg", "jena_000037_000019_heavyfog_v0_beta0.142.jpg"),
    ("krefeld", "krefeld_000000_026269", "krefeld_000000_026269_lightfog_v0_beta0.035.jpg", "krefeld_000000_026269_mediumfog_v0_beta0.066.jpg", "krefeld_000000_026269_heavyfog_v0_beta0.156.jpg"),
    ("monchengladbach", "monchengladbach_000000_013228", "monchengladbach_000000_013228_lightfog_v0_beta0.030.jpg", "monchengladbach_000000_013228_mediumfog_v0_beta0.087.jpg", "monchengladbach_000000_013228_heavyfog_v0_beta0.158.jpg"),
]

THUMB_W = 512
THUMB_H = 256
PAD = 4
LABEL_H = 32
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.65
FONT_THICK = 2

def put_label(canvas, text, x, y, w):
    """Center text at (x, y) within width w."""
    sz = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)[0]
    tx = x + (w - sz[0]) // 2
    cv2.putText(canvas, text, (tx, y), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK)

for city, base_name, light_f, medium_f, heavy_f in SCENES:
    paths = [
        ("Clear", os.path.join(BASE, "images_clear", base_name + ".png")),
        ("Light Fog", os.path.join(BASE, "images_foggy/light", light_f)),
        ("Medium Fog", os.path.join(BASE, "images_foggy/medium", medium_f)),
        ("Heavy Fog", os.path.join(BASE, "images_foggy/heavy", heavy_f)),
    ]

    imgs = []
    for label, p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  ERROR reading {p}")
            break
        imgs.append(cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA))

    if len(imgs) != 4:
        print(f"Skipping {city}")
        continue

    # 2x2 grid layout
    # Row 1: Clear | Light
    # Row 2: Medium | Heavy
    grid_w = 2 * THUMB_W + 3 * PAD
    grid_h = 2 * (LABEL_H + THUMB_H) + 3 * PAD
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    labels = ["Clear", "Light Fog", "Medium Fog", "Heavy Fog"]
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (col, row)

    for i, (col, row) in enumerate(positions):
        x = PAD + col * (THUMB_W + PAD)
        y_label = PAD + row * (LABEL_H + THUMB_H + PAD)
        y_img = y_label + LABEL_H

        put_label(canvas, labels[i], x, y_label + LABEL_H - 8, THUMB_W)
        canvas[y_img:y_img + THUMB_H, x:x + THUMB_W] = imgs[i]

    out_path = os.path.join(OUT_DIR, f"grid2x2_{city}.jpg")
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out_path}")

print("\nDone! 10 grids saved to debug/")
