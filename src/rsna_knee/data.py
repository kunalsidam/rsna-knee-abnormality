"""Metadata loading and the multi-plane DICOM dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import DEFAULT_IMG_SIZE, PLANES, TARGET_COLS

log = logging.getLogger(__name__)


def load_metadata(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read the competition CSVs and check the target columns line up."""
    data_dir = Path(data_dir)
    tables = {
        "train": pd.read_csv(data_dir / "train.csv"),
        "train_series": pd.read_csv(data_dir / "train_series.csv"),
        "test": pd.read_csv(data_dir / "test.csv"),
        "test_series": pd.read_csv(data_dir / "test_series.csv"),
        "sample_submission": pd.read_csv(data_dir / "sample_submission.csv"),
    }
    sub_cols = list(tables["sample_submission"].columns[1:])
    if sub_cols != TARGET_COLS:
        raise ValueError(
            f"Target columns in sample_submission.csv differ from config.TARGET_COLS:\n{sub_cols}"
        )
    return tables


def gold_mask(train_df: pd.DataFrame) -> pd.Series:
    """Studies where all 12 expert labels are present (58 in the local copy)."""
    return train_df[TARGET_COLS].notna().all(axis=1)


class KneeDataset(Dataset):
    """One sample = one study, represented as a 3-channel image.

    Channel 0 = middle slice of a Sagittal series, 1 = Coronal, 2 = Axial.
    A missing plane or unreadable file becomes an all-zero channel (and is logged).

    Items are dicts: {"uid", "image"} plus "target" when labels are given.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        series_df: pd.DataFrame,
        series_root: str | Path,
        labels: np.ndarray | None = None,
        img_size: int = DEFAULT_IMG_SIZE,
    ):
        self.uids = df["StudyInstanceUID"].tolist()
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.series_root = Path(series_root)
        self.img_size = img_size

        # Index once: study -> {plane: first SeriesInstanceUID}. The notebook filtered
        # the whole series table inside __getitem__, which is slow.
        self.series_by_study: dict[str, dict[str, str]] = {}
        for (uid, plane), grp in series_df.groupby(["StudyInstanceUID", "Anatomical_Plane"]):
            self.series_by_study.setdefault(uid, {})[plane] = grp["SeriesInstanceUID"].iloc[0]

    def __len__(self) -> int:
        return len(self.uids)

    def _zeros(self) -> torch.Tensor:
        return torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

    def _load_middle_slice(self, study_uid: str, series_uid: str) -> torch.Tensor:
        import pydicom  # imported here so the module loads without it

        series_dir = self.series_root / study_uid / series_uid
        try:
            files = sorted(p for p in series_dir.iterdir() if p.suffix == ".dcm")
            if not files:
                raise FileNotFoundError(f"no .dcm files in {series_dir}")
            img = pydicom.dcmread(files[len(files) // 2]).pixel_array.astype(np.float32)
            peak = img.max()
            if peak > 0:
                img = img / peak
            t = torch.from_numpy(img)[None, None]  # [1, 1, H, W]
            t = F.interpolate(t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            return t[0]  # [1, H, W]
        except Exception as exc:  # keep training alive, but never fail silently
            log.warning("Could not read %s/%s: %s", study_uid, series_uid, exc)
            return self._zeros()

    def __getitem__(self, idx: int) -> dict:
        uid = self.uids[idx]
        planes = self.series_by_study.get(uid, {})
        channels = []
        for plane in PLANES:
            series_uid = planes.get(plane)
            channels.append(self._load_middle_slice(uid, series_uid) if series_uid else self._zeros())
        item = {"uid": uid, "image": torch.cat(channels, dim=0)}  # [3, H, W]
        if self.labels is not None:
            item["target"] = torch.from_numpy(self.labels[idx])
        return item
