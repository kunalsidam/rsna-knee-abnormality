"""Heuristic soft labels parsed from the free-text radiology reports.

Only ~58 of 4,407 training studies have expert labels, but every study has a
report. This module turns each report into 12 rough probabilities.

Differences from the notebook version: keywords match on word boundaries and
negation is checked *within the same sentence* as the keyword. The notebook
looked for negation words anywhere in the report, so a single "normal" elsewhere
down-weighted every finding.

Usage:
    python -m rsna_knee.labels --data-dir /path/to/competition --out heuristic_train_labels.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .config import TARGET_COLS

KEYWORD_RULES: dict[str, list[str]] = {
    "ACL": ["acl", "anterior cruciate ligament"],
    "MCL": ["mcl", "medial collateral ligament"],
    "Medial Meniscus": ["medial meniscus", "medial meniscal"],
    "Lateral Meniscus": ["lateral meniscus", "lateral meniscal"],
    "Medial OA": ["medial compartment", "medial femorotibial", "medial osteophytes"],
    "Lateral OA": ["lateral compartment", "lateral femorotibial", "lateral osteophytes"],
    "PF OA": ["patellofemoral", "patella", "trochlea", "chondromalacia"],
    "Effusion": ["effusion", "fluid"],
    "Synovitis": ["synovitis", "synovial"],
    "Baker's": ["baker's", "popliteal cyst"],
    "Contusion": ["contusion", "bone bruise", "marrow edema"],
    "Fracture": ["fracture", "break"],
}

NEGATION = re.compile(
    r"\b(intact|normal|unremarkable|preserved|no|without|negative for|absent)\b", re.IGNORECASE
)
SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+|\n+")

P_POSITIVE = 0.8  # keyword present in a non-negated sentence
P_NEGATED = 0.3   # keyword only appears in negated sentences
P_ABSENT = 0.1    # keyword never appears


def _compile(keywords: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b", re.IGNORECASE)


_PATTERNS = {col: _compile(kws) for col, kws in KEYWORD_RULES.items()}


def score_report(report: str | float | None) -> dict[str, float]:
    """Return {target: soft probability} for one report."""
    text = "" if report is None or (isinstance(report, float) and pd.isna(report)) else str(report)
    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    scores: dict[str, float] = {}
    for col in TARGET_COLS:
        pattern = _PATTERNS[col]
        hits = [s for s in sentences if pattern.search(s)]
        if not hits:
            scores[col] = P_ABSENT
        elif any(not NEGATION.search(s) for s in hits):
            scores[col] = P_POSITIVE
        else:
            scores[col] = P_NEGATED
    return scores


def generate_heuristic_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    """One row per study: StudyInstanceUID + 12 soft-label columns."""
    rows = [score_report(r) for r in train_df["Report"]]
    out = pd.DataFrame(rows, columns=TARGET_COLS)
    out.insert(0, "StudyInstanceUID", train_df["StudyInstanceUID"].values)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", default="heuristic_train_labels.csv", type=Path)
    args = ap.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    labels = generate_heuristic_labels(train_df)
    labels.to_csv(args.out, index=False)
    print(f"Wrote {len(labels)} rows x {len(TARGET_COLS)} targets to {args.out}")


if __name__ == "__main__":
    main()
