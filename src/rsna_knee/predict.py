"""Run a trained checkpoint on the test studies and write submission.csv.

Example:
    python -m rsna_knee.predict --data-dir /kaggle/input/competitions/rsna-knee-abnormality-detection \
        --checkpoint outputs/best.pth --out submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TARGET_COLS
from .data import KneeDataset, load_metadata
from .model import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", default=Path("submission.csv"), type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    tables = load_metadata(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    ds = KneeDataset(tables["test"], tables["test_series"], args.data_dir / "test_series", None, args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"].to(device))).cpu().numpy()
            for uid, p in zip(batch["uid"], probs):
                rows.append({"StudyInstanceUID": uid, **{c: float(v) for c, v in zip(TARGET_COLS, p)}})

    sub = pd.DataFrame(rows)[["StudyInstanceUID"] + TARGET_COLS]
    sub.to_csv(args.out, index=False)
    print(f"wrote {len(sub)} rows to {args.out}")


if __name__ == "__main__":
    main()
