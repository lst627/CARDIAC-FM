"""
Dataset for multi-view cardiac MRI contrastive pretraining.

Supports two backends:
  - HDF5  (.h5 file)  : fast on network filesystems — recommended for /chru/data/
  - numpy (.npy files): one file per subject — fast on local /scratch/

Each sample is one subject with three views:
  vst_2ch  →  (1,  25, 224, 224)  float32
  vst_4ch  →  (1,  25, 224, 224)  float32
  vst_sa   →  (3,  25, 224, 224)  float32
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CardiacMRIDataset(Dataset):
    def __init__(self, data_root: str, augment: bool = True,
                 subject_list=None, hdf5_path: str = None,
                 spatial_augment: bool = False, img_size: int = None):
        """
        Parameters
        ----------
        data_root        : directory with per-subject .npy folders (used when hdf5_path is None)
        augment          : random temporal crop during training; center crop otherwise
        subject_list     : optional collection of EID strings to restrict to
        hdf5_path        : path to packed HDF5 file; if provided, reads from HDF5 instead of .npy
        spatial_augment  : apply spatial/intensity augmentations (flip, crop, noise, brightness)
        img_size         : if set, resize spatial dims to (img_size, img_size) after loading
        """
        self.augment         = augment
        self.spatial_augment = spatial_augment
        self.hdf5_path       = hdf5_path
        self.img_size        = img_size
        self._hdf5_file = None   # opened lazily per worker

        allowed = set(subject_list) if subject_list is not None else None

        if hdf5_path is not None:
            import h5py
            with h5py.File(hdf5_path, "r") as f:
                all_keys = list(f.keys())
            if allowed is not None:
                self.subjects = [k for k in sorted(all_keys) if k in allowed]
            else:
                self.subjects = sorted(all_keys)
        else:
            data_root = Path(data_root)
            self.subjects = []
            for d in sorted(data_root.iterdir()):
                if not d.is_dir():
                    continue
                if allowed is not None and d.name not in allowed:
                    continue
                if all((d / f).exists()
                       for f in ["vst_2ch.npy", "vst_4ch.npy", "vst_sa.npy"]):
                    self.subjects.append(str(d))

        if len(self.subjects) == 0:
            raise RuntimeError("No valid subjects found")

    def __len__(self):
        return len(self.subjects)

    def _get_hdf5(self):
        # open HDF5 lazily — each worker gets its own file handle
        if self._hdf5_file is None:
            import h5py
            self._hdf5_file = h5py.File(self.hdf5_path, "r")
        return self._hdf5_file

    def _load_hdf5(self, eid):
        f = self._get_hdf5()
        v2ch = torch.from_numpy(f[eid]["vst_2ch"][:].astype(np.float32))
        v4ch = torch.from_numpy(f[eid]["vst_4ch"][:].astype(np.float32))
        vsa  = torch.from_numpy(f[eid]["vst_sa"][:].astype(np.float32))
        return v2ch, v4ch, vsa

    def _load_npy(self, subj_path):
        p = Path(subj_path)
        v2ch = torch.from_numpy(np.load(str(p / "vst_2ch.npy")).astype(np.float32))
        v4ch = torch.from_numpy(np.load(str(p / "vst_4ch.npy")).astype(np.float32))
        vsa  = torch.from_numpy(np.load(str(p / "vst_sa.npy")).astype(np.float32))
        return v2ch, v4ch, vsa

    def _crop_frames(self, x: torch.Tensor, n_frames: int = 8) -> torch.Tensor:
        T = x.shape[1]
        if T <= n_frames:
            return x
        start = random.randint(0, T - n_frames) if self.augment \
                else (T - n_frames) // 2
        return x[:, start:start + n_frames]

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize spatial dims to img_size × img_size. Input: (C, T, H, W)"""
        C, T, H, W = x.shape
        if H == self.img_size and W == self.img_size:
            return x
        x = x.view(C * T, 1, H, W)
        x = F.interpolate(x, size=(self.img_size, self.img_size),
                          mode="bilinear", align_corners=False)
        return x.view(C, T, self.img_size, self.img_size)

    def _spatial_aug(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial/intensity augmentations consistently across all frames.
        Input shape: (C, T, H, W)
        """
        C, T, H, W = x.shape

        # random horizontal flip
        if random.random() < 0.5:
            x = x.flip(-1)

        # random crop + resize (zoom): same crop for all frames
        scale = random.uniform(0.85, 1.0)
        ch = int(H * scale)
        cw = int(W * scale)
        top  = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        x = x[:, :, top:top + ch, left:left + cw]
        # resize back to original spatial size
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        # random intensity scaling (per-volume, not per-frame)
        scale_i = random.uniform(0.8, 1.2)
        x = x * scale_i

        # gaussian noise
        x = x + torch.randn_like(x) * 0.02

        return x

    def __getitem__(self, idx):
        subj = self.subjects[idx]

        if self.hdf5_path is not None:
            eid = subj
            v2ch, v4ch, vsa = self._load_hdf5(eid)
        else:
            eid = Path(subj).name
            v2ch, v4ch, vsa = self._load_npy(subj)

        if self.img_size is not None:
            v2ch = self._resize(v2ch)
            v4ch = self._resize(v4ch)
            vsa  = self._resize(vsa)

        if self.spatial_augment:
            v2ch = self._spatial_aug(v2ch)
            v4ch = self._spatial_aug(v4ch)
            vsa  = self._spatial_aug(vsa)

        return {
            "v2ch":    self._crop_frames(v2ch),
            "v4ch":    self._crop_frames(v4ch),
            "vsa":     self._crop_frames(vsa),
            "subject": eid,
        }
