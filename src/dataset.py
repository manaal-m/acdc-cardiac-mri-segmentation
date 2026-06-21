"""
ACDC dataset loading, preprocessing, and PyTorch Lightning DataModule.

Handles the MICCAI 2017 Automated Cardiac Diagnosis Challenge (ACDC) data
layout: one folder per patient, each containing an Info.cfg file (with ED/ES
frame indices and disease group) plus NIfTI volumes for the ED and ES frames
and their ground-truth segmentations.
"""

import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES = {
    0: "background",
    1: "LV cavity",
    2: "RV cavity",
    3: "Myocardium",
}

TARGET_H, TARGET_W = 512, 512


def load_nii(path):
    """Load a NIfTI file and return its data array."""
    return nib.load(path).get_fdata()


def load_info_cfg(info_path):
    """Parse an ACDC Info.cfg file into a dict, e.g. {'ED': '1', 'ES': '14', 'Group': 'MINF'}."""
    info = {}
    with open(info_path, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":")
                info[k.strip()] = v.strip()
    return info


def get_acdc_cases(root):
    """Walk the ACDC training root and return one record per patient."""
    patients = []
    for patient in sorted(os.listdir(root)):
        pdir = os.path.join(root, patient)
        if not os.path.isdir(pdir):
            continue
        info = load_info_cfg(os.path.join(pdir, "Info.cfg"))
        patients.append(
            {
                "patient": patient,
                "pdir": pdir,
                "ED": int(info["ED"]),
                "ES": int(info["ES"]),
                "group": info["Group"],
            }
        )
    return patients


def patient_to_cases(patients):
    """Expand each patient into ED and ES frame cases (image + label paths)."""
    cases = []
    for p in patients:
        for frame in [p["ED"], p["ES"]]:
            img = os.path.join(p["pdir"], f"{p['patient']}_frame{frame:02d}.nii")
            lbl = os.path.join(p["pdir"], f"{p['patient']}_frame{frame:02d}_gt.nii")
            cases.append({"image": img, "label": lbl, "group": p["group"]})
    return cases


def normalize(img):
    """Z-score normalize an image array."""
    return (img - img.mean()) / (img.std() + 1e-5)


def _center_pad_or_crop(arr, target_h, target_w):
    """Pad (with zeros) or center-crop a 2D array to (target_h, target_w)."""
    h, w = arr.shape

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")

    if arr.shape[0] > target_h:
        start = (arr.shape[0] - target_h) // 2
        arr = arr[start : start + target_h, :]
    if arr.shape[1] > target_w:
        start = (arr.shape[1] - target_w) // 2
        arr = arr[:, start : start + target_w]

    return arr


class ACDC2DDataset(Dataset):
    """2D slice dataset: every axial slice of every ED/ES volume becomes one sample.

    Each sample is z-score normalized and center-padded/cropped to
    (TARGET_H, TARGET_W) so a batch can be formed even though raw ACDC
    slices vary in size across patients and scanners.
    """

    def __init__(self, cases):
        self.samples = []
        for item in cases:
            img = load_nii(item["image"])
            mask = load_nii(item["label"])
            for d in range(img.shape[2]):
                self.samples.append({"image": img[:, :, d], "mask": mask[:, :, d]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = s["image"].astype(np.float32)
        mask = s["mask"].astype(np.int64)

        img = normalize(img)
        img = _center_pad_or_crop(img, TARGET_H, TARGET_W)
        mask = _center_pad_or_crop(mask, TARGET_H, TARGET_W)

        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask)
        return img, mask


class ACDC2DDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping the ACDC2DDataset with a patient-level,
    group-stratified train/val/test split (70/15/15).

    The split is stratified on disease group (NOR, MINF, DCM, HCM, RV) and is
    done at the PATIENT level, then expanded to slices, so no slices from the
    same patient leak across splits.
    """

    def __init__(self, root, batch_size=8, num_workers=2, seed=42):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        patients = get_acdc_cases(self.root)
        groups = [p["group"] for p in patients]

        train_patients, val_test_patients = train_test_split(
            patients, test_size=0.3, stratify=groups, random_state=self.seed
        )
        val_test_groups = [p["group"] for p in val_test_patients]
        test_patients, val_patients = train_test_split(
            val_test_patients, test_size=0.5, stratify=val_test_groups, random_state=self.seed
        )

        self.train_ds = ACDC2DDataset(patient_to_cases(train_patients))
        self.val_ds = ACDC2DDataset(patient_to_cases(val_patients))
        self.test_ds = ACDC2DDataset(patient_to_cases(test_patients))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
