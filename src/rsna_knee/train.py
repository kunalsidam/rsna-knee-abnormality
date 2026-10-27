"""Train the baseline and save the best checkpoint.

Two label sources:
  gold       train on the expert-labelled studies (random 80/20 split).
  heuristic  train on report-derived soft labels for every non-gold study and
             validate on ALL gold studies, which keeps validation labels real.

Example:
    python -m rsna_knee.train --data-dir /kaggle/input/competitions/rsna-knee-abnormality-detection \
        --label-source heuristic --epochs 5 --out-dir outputs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import TARGET_COLS
from .data import KneeDataset, gold_mask, load_metadata
from .labels import generate_heuristic_labels
from .model import build_model


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    """Mean ROC-AUC over targets that have both classes present."""
    per_class = {}
    for i, col in enumerate(TARGET_COLS):
        if len(np.unique(y_true[:, i])) > 1:
            per_class[col] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
    mean = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return mean, per_class


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs, targets = [], []
    for batch in loader:
        out = model(batch["image"].to(device))
        probs.append(torch.sigmoid(out).cpu().numpy())
        targets.append(batch["target"].numpy())
    return macro_auc(np.concatenate(targets), np.concatenate(probs))


def make_splits(train_df, label_source: str, val_frac: float, seed: int):
    """Return (train_df, train_labels, val_df, val_labels)."""
    is_gold = gold_mask(train_df)
    gold_df = train_df[is_gold].reset_index(drop=True)

    if label_source == "gold":
        tr, va = train_test_split(np.arange(len(gold_df)), test_size=val_frac, random_state=seed)
        y = gold_df[TARGET_COLS].values.astype(np.float32)
        return gold_df.iloc[tr], y[tr], gold_df.iloc[va], y[va]

    heur = generate_heuristic_labels(train_df)
    rest = train_df[~is_gold].reset_index(drop=True)
    rest_labels = heur.set_index("StudyInstanceUID").loc[rest["StudyInstanceUID"], TARGET_COLS].values
    return rest, rest_labels.astype(np.float32), gold_df, gold_df[TARGET_COLS].values.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("outputs"), type=Path)
    ap.add_argument("--label-source", choices=["gold", "heuristic"], default="heuristic")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2, help="only used with --label-source gold")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_metadata(args.data_dir)
    tr_df, tr_y, va_df, va_y = make_splits(tables["train"], args.label_source, args.val_frac, args.seed)
    print(f"train studies: {len(tr_df)} | validation studies: {len(va_df)} | labels: {args.label_source}")

    series_root = args.data_dir / "train_series"
    train_ds = KneeDataset(tr_df, tables["train_series"], series_root, tr_y, args.img_size)
    val_ds = KneeDataset(va_df, tables["train_series"], series_root, va_y, args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()  # accepts soft targets in [0, 1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            images, targets = batch["image"].to(device), batch["target"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_ds)

        auc, per_class = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_auc": auc, "per_class_auc": per_class})
        print(f"epoch {epoch}/{args.epochs} | loss {train_loss:.4f} | val macro AUC {auc:.4f} ({len(per_class)}/12 classes)")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.out_dir / "best.pth")

    if best_auc < 0:  # AUC never defined; still save something predict.py can load
        torch.save(model.state_dict(), args.out_dir / "best.pth")
    (args.out_dir / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"saved {args.out_dir / 'best.pth'} and metrics.json")


if __name__ == "__main__":
    main()
