import os
import shutil

IMG_ROOT = "raw_cityscapes/leftImg8bit/train"
OUTPUT_DIR = "images_clear"

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0
for city in sorted(os.listdir(IMG_ROOT)):
    city_path = os.path.join(IMG_ROOT, city)
    for f in sorted(os.listdir(city_path)):
        if not f.endswith(".png"):
            continue
        base = f.replace("_leftImg8bit.png", "")
        dst = os.path.join(OUTPUT_DIR, base + ".png")
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(city_path, f), dst)
        count += 1

print(f"Total clear images in {OUTPUT_DIR}: {count}")
