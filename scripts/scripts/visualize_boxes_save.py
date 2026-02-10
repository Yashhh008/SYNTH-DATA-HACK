import cv2
import os

IMG_ROOT = "raw_cityscapes/leftImg8bit/train"
LABEL_DIR = "annotations"
DEBUG_DIR = "debug"

os.makedirs(DEBUG_DIR, exist_ok=True)

CLASS_NAMES = {
    0: "person",
    1: "car",
    2: "bicycle",
    3: "motorcycle"
}

# pick first 5 annotation files
label_files = sorted(os.listdir(LABEL_DIR))[:5]

for label_file in label_files:
    base = label_file.replace(".txt", "")

    # find image (search across cities)
    img_path = None
    for city in os.listdir(IMG_ROOT):
        candidate = os.path.join(
            IMG_ROOT, city, base + "_leftImg8bit.png"
        )
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        continue

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    with open(os.path.join(LABEL_DIR, label_file)) as f:
        lines = f.readlines()

    for line in lines:
        class_id, xc, yc, w, h = map(float, line.split())

        x_center = int(xc * W)
        y_center = int(yc * H)
        bw = int(w * W)
        bh = int(h * H)

        x1 = x_center - bw // 2
        y1 = y_center - bh // 2
        x2 = x_center + bw // 2
        y2 = y_center + bh // 2

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            CLASS_NAMES[int(class_id)],
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    out_path = os.path.join(DEBUG_DIR, base + "_debug.jpg")
    cv2.imwrite(out_path, img)

print("Debug images saved in /debug folder")
