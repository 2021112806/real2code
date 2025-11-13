import os
import argparse
from glob import glob
from natsort import natsorted
import subprocess

"""
Invoke DUSt3R CLI to generate per-image *_scene.npz pointmaps for a folder,
matching the layout in datasets/real_world_data/8.

Requirements:
- The `dust3r` package with its CLI available:
    python -m dust3r.inference --help

Usage:
  python -m real_obj.gen_dust3r_npz \
    --data_dir /mnt/data/zhangzhaodong/real2code/datasets/real_world_data \
    --folder 8 \
    --pattern "*.jpg" \
    --extra "--save-npz"

This will call DUSt3R to reconstruct the set of images in that folder and
save *_scene.npz files alongside the images.
"""


def ensure_dust3r():
    try:
        out = subprocess.run(
            ["python", "-m", "dust3r.inference", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return out.returncode in (0, 2)  # help may return 2
    except Exception:
        return False


def run_dust3r_on_folder(root: str, folder: str, pattern: str, extra: str):
    obj_dir = os.path.join(root, folder)
    imgs = natsorted(glob(os.path.join(obj_dir, pattern)))
    if len(imgs) == 0:
        print(f"No images matched {obj_dir}/{pattern}")
        return
    cmd = [
        "python", "-m", "dust3r.inference",
        "--images", os.path.join(obj_dir, pattern),
        "--outdir", obj_dir,
    ]
    if extra:
        cmd += extra.split()
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument("--extra", default="--save-npz")
    args = ap.parse_args()

    if not ensure_dust3r():
        raise RuntimeError(
            "dust3r CLI not found. Please install dust3r and ensure 'python -m dust3r.inference --help' works."
        )
    run_dust3r_on_folder(args.data_dir, args.folder, args.pattern, args.extra)


if __name__ == "__main__":
    main()


