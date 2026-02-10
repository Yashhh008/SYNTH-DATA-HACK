import os
import cv2
import numpy as np

# ==============================
# CONFIG
# ==============================

GT_FINE_ROOT = "raw_cityscapes/gtFine/train"
IMG_ROOT = "raw_cityscapes/leftImg8bit/train"
OUTPUT_DIR = "annotations"

MIN_BOX_SIZE = 10  # pixels

# Cityscapes classId → YOLO class_id
CITYSCAPES_TO_YOLO = {
    24: 0,  # person
    26: 1,  # car
    33: 2,  # bicycle
    32: 3   # motorcycle
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# INSTANCE-BASED BBOX EXTRACTION
# ==============================

def extract_bboxes_from_instances(instance_mask, target_class_id):
    """
    Extract bounding boxes for each object instance
    of a given Cityscapes class ID.
    """
    bboxes = []

    instance_ids = np.unique(instance_mask)

    for inst_id in instance_ids:
        if inst_id < 1000:
            continue  # background / stuff

        class_id = inst_id // 1000
        if class_id != target_class_id:
            continue

        ys, xs = np.where(instance_mask == inst_id)
        if len(xs) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        if (x_max - x_min) < MIN_BOX_SIZE or (y_max - y_min) < MIN_BOX_SIZE:
            continue

        bboxes.append((x_min, y_min, x_max, y_max))

    return bboxes


# ==============================
# MAIN LOOP
# ==============================

for city in os.listdir(GT_FINE_ROOT):
    city_gt_path = os.path.join(GT_FINE_ROOT, city)
    city_img_path = os.path.join(IMG_ROOT, city)

    for file in os.listdir(city_gt_path):

        # ✅ USE INSTANCE IDS (NOT labelIds)
        if not file.endswith("_gtFine_instanceIds.png"):
            continue

        instance_path = os.path.join(city_gt_path, file)

        base_name = file.replace("_gtFine_instanceIds.png", "")
        img_name = base_name + "_leftImg8bit.png"
        img_path = os.path.join(city_img_path, img_name)

        if not os.path.exists(img_path):
            continue

        instance_mask = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(img_path)

        if img is None or instance_mask is None:
            continue

        H, W = img.shape[:2]
        yolo_lines = []

        for cityscapes_id, yolo_id in CITYSCAPES_TO_YOLO.items():
            bboxes = extract_bboxes_from_instances(instance_mask, cityscapes_id)

            for (x_min, y_min, x_max, y_max) in bboxes:
                x_center = ((x_min + x_max) / 2) / W
                y_center = ((y_min + y_max) / 2) / H
                box_width = (x_max - x_min) / W
                box_height = (y_max - y_min) / H

                yolo_lines.append(
                    f"{yolo_id} {x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}"
                )

        if yolo_lines:
            out_file = os.path.join(OUTPUT_DIR, base_name + ".txt")
            with open(out_file, "w") as f:
                f.write("\n".join(yolo_lines))

print("✅ YOLO annotation generation (instance-based) completed.")
