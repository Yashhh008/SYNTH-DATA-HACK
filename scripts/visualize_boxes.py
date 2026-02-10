import cv2
import os

IMG_DIR = "raw_cityscapes/leftImg8bit/train/frankfurt"
LABEL_DIR = "annotations"

# pick any 5 annotation files
sample_files = os.listdir(LABEL_DIR)[:5]

CLASS_NAMES = {
    0: "person",
    1: "car",
    2: "bicycle",
    3: "motorcycle"
}

for label_file in sample_files:
    img_name = label_file.replace(".txt", "_leftImg8bit.png")
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
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
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imshow("Bounding Box Check", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
