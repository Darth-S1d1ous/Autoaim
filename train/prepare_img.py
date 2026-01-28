import os
import glob
import shutil

splits = ["train", "test"]

image_root = "D:/GTA_Head/GTA_Head"
dst_root = "D:/Autoaim/train/data/images"
train_path = "data/train.txt"
test_path = "data/test.txt"  

if os.path.exists(train_path):
    os.remove(train_path)
if os.path.exists(test_path):
    os.remove(test_path)

for split in splits:
    for img_path in glob.glob(os.path.join(image_root, split, "scene*", "img1", "*.jpg")):
        scene_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        print(scene_name)

        dst_dir = os.path.join(dst_root, split, scene_name)
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(img_path))

        shutil.copyfile(img_path, dst_path)

with open(train_path, "w") as out_f:
    for img in sorted(glob.glob(os.path.join(dst_root, "train", "scene*", "*.jpg"))):
        out_f.write(img.replace("\\", "/") + '\n')

with open(test_path, "w") as out_f:
    for img in sorted(glob.glob(os.path.join(dst_root, "test", "scene*", "*.jpg"))):
        out_f.write(img.replace("\\", "/") + '\n')