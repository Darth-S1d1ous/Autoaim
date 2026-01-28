import os
import glob
from tqdm import tqdm

W, H = 1920, 1080

splits = ["train", "test"]

dataset_dir = "D:/GTA_Head/GTA_Head"
out_dir = "data/labels"
os.makedirs(out_dir, exist_ok=True)

for split in splits:
    scene_dirs = sorted(glob.glob(os.path.join(dataset_dir, split, "scene*")))

    for scene_dir in tqdm(scene_dirs, desc=f"Processing {split} scenes"):
        gt_path = os.path.join(scene_dir, "gt", "gt.txt")
        if not os.path.isfile(gt_path):
            print(f"Warning: {gt_path} not found, skipping.")
            continue
        # build output dir
        out_scene_dir = os.path.join(out_dir, split, os.path.basename(scene_dir))
        os.makedirs(out_scene_dir, exist_ok=True)

        frame_dict = {}

        with open(gt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(' ')
                frame_id = int(float(items[0]))
                x, y, w, h = map(float, items[2:6])

                xc = (x + w / 2) / W
                yc = (y + h / 2) / H

                bw = w / W
                bh = h / H

                yolo_line = f"0 {xc:6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

                frame_dict.setdefault(frame_id, []).append(yolo_line)

        for frame_id, lines in frame_dict.items():
            with open(f"{out_scene_dir}/{frame_id:06d}.txt", "w") as f:
                f.writelines(lines)