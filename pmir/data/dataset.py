import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from pmir.data.augmentation import augment_polar_angles
from pmir.data.polarization import compute_dop
from pmir.data.image_io import load_image_rgb


class PolarDataset(Dataset):
    ANGLES = ("0", "45", "90", "135")
    SCATTER_DIRS = {a: f"scatter_{a}" for a in ANGLES}
    RAW_DIRS = {a: f"raw_{a}" for a in ANGLES}

    def __init__(self, scatter_root, raw_root, size=256, augment=False):
        self.size = size
        self.augment = augment
        self.samples = []
        sc0_dir = os.path.join(scatter_root, self.SCATTER_DIRS["0"])
        if not os.path.isdir(sc0_dir):
            raise RuntimeError(f"Cannot find scatter_0 directory: {sc0_dir}")

        incomplete = []
        for filename in sorted(os.listdir(sc0_dir)):
            if not filename.lower().endswith(".bmp"):
                continue
            stem = os.path.splitext(filename)[0]
            if not stem.endswith("_0"):
                continue
            base = stem[:-2]
            sc = {
                a: os.path.join(scatter_root, self.SCATTER_DIRS[a], f"{base}_{a}.bmp")
                for a in self.ANGLES
            }
            raw = {
                a: os.path.join(raw_root, self.RAW_DIRS[a], f"{base}_{a}.bmp")
                for a in self.ANGLES
            }
            missing = [p for p in list(sc.values()) + list(raw.values()) if not os.path.exists(p)]
            if missing:
                incomplete.append((base, missing))
                continue
            self.samples.append((sc, raw))

        if incomplete:
            print(f"[dataset] skipped {len(incomplete)} incomplete four-angle samples")
        if not self.samples:
            raise RuntimeError("No complete scatter/raw four-angle pairs were found.")
        print(f"[dataset] complete four-angle samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sc_paths, raw_paths = self.samples[index]
        sc = {a: load_image_rgb(sc_paths[a], self.size) for a in self.ANGLES}
        raw = {a: load_image_rgb(raw_paths[a], self.size) for a in self.ANGLES}
        if self.augment:
            code = random.randint(0, 5)
            sc = augment_polar_angles(sc, code)
            raw = augment_polar_angles(raw, code)

        dop = compute_dop(sc["0"], sc["45"], sc["90"], sc["135"])

        angle_input = np.concatenate([sc[a] for a in self.ANGLES], axis=2)
        gt_angle = np.concatenate([raw[a] for a in self.ANGLES], axis=2)
        angle_input = torch.from_numpy(angle_input.transpose(2, 0, 1)).float()
        gt_angle = torch.from_numpy(gt_angle.transpose(2, 0, 1)).float()
        return angle_input, dop, gt_angle
