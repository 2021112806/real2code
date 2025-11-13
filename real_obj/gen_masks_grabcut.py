import os
import argparse
from glob import glob
from natsort import natsorted
import numpy as np
import cv2


"""
Generate binary foreground masks for real_world_data folders using OpenCV GrabCut.

Output mask file per image: <name>_mask.png (white=foreground, black=background)
Placed in the SAME directory as the source image, matching the structure
seen in datasets/real_world_data/8.

Usage:
  python -m real_obj.gen_masks_grabcut \
    --data_dir /mnt/data/zhangzhaodong/real2code/datasets/real_world_data \
    --folder 8

Notes:
- This is a classical baseline requiring only OpenCV. If you have a trained SAM, 
  replace this with your SAM inference for better quality.
"""


def generate_mask_grabcut(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    # initial rectangle slightly inset from image border
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # Convert to binary: probable/definite foreground -> 1
    bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Clean up
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    bin_mask = cv2.medianBlur(bin_mask, 5)
    return bin_mask


def process_folder(root: str, folder: str):
    obj_dir = os.path.join(root, folder)
    jpgs = natsorted(glob(os.path.join(obj_dir, "*.jpg")))
    print(f"Found {len(jpgs)} images in {obj_dir}")
    for img_path in jpgs:
        mask_path = img_path.replace(".jpg", "_mask.png")
        if os.path.exists(mask_path):
            continue
        rgb = cv2.imread(img_path)
        if rgb is None:
            print(f"WARN: cannot read {img_path}, skipping")
            continue
        mask = generate_mask_grabcut(rgb)
        cv2.imwrite(mask_path, mask)
        print(f"Wrote {mask_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--folder", required=True)
    args = ap.parse_args()
    process_folder(args.data_dir, args.folder)


if __name__ == "__main__":
    main()


