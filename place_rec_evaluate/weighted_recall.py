#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute weighted mean recall per model.

- Input 1 (required): recall CSV with per-sequence results. Must include a "model" column and
  several recall columns (e.g., R@1, R@5, R@10) and one or more identifier columns that define a sequence
  (e.g., dataset, dataset_splits, q/db, sequence, etc.).
- Input 2 (optional): frames CSV with the number of frames (queries) per sequence. If the recall CSV lacks
  a frames/weight column, this file is required.
  It should contain the same identifier columns as the recall CSV for a merge, plus a single frames column.

Weighted mean for each recall column is:
    sum(recall_i * frames_i) / sum(frames_i), grouped by model.

Usage:
  python compute_weighted_recalls.py \
      --recall_csv recall_results_RGB_THERMAL_same_backbone.csv \
      --frames_csv frames_per_sequence.csv \
      --out_csv weighted_recall_means.csv
"""
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List

MODEL_CANDIDATES = [
    "model"
]

WEIGHT_CANDIDATES = [
    "q/db"
]

# plausible identifier (join) columns that define a sequence
ID_CANDIDATES = [
    "dataset_splits", "dataset_splits or sequence"
]

def find_first(columns: List[str], candidates: List[str]):
    return next((c for c in candidates if c in columns), None)

def is_recall_col(c: str) -> bool:
    cl = c.lower()
    return (
        ("recall" in cl) 
        or ("r@" in cl)
        or bool(re.fullmatch(r"r\d+", cl))
        or bool(re.fullmatch(r"recall@\d+", cl))
        or bool(re.fullmatch(r"recall_\d+", cl))
    )

def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(series[mask], weights=weights[mask]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall_csv", required=True, type=Path)
    ap.add_argument("--out_csv", required=False, type=Path, default=Path("weighted_recall_means.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.recall_csv)
    columns = list(df.columns)

    model_col = find_first(columns, MODEL_CANDIDATES)
    if model_col is None:
        raise SystemExit(f"Could not find a model column. Looked for: {MODEL_CANDIDATES}\nAvailable: {columns}")

    # detect recall columns (numeric)
    recall_cols = [c for c in columns if is_recall_col(c)]
    for c in recall_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    recall_cols = [c for c in recall_cols if df[c].notna().any()]
    if not recall_cols:
        raise SystemExit(f"Could not find any recall columns. Available: {columns}")

    # get or merge weights
    weight_col = find_first(columns, WEIGHT_CANDIDATES)
    # if weight_col is None:
    #     if args.frames_csv is None:
    #         # Try to infer join keys and instruct the user
    #         present_ids = [c for c in ID_CANDIDATES if c in columns]
    #         raise SystemExit(
    #             "No frames/weight column found in recall CSV and --frames_csv not provided.\n"
    #             f"Add a frames column to the recall CSV (any of {WEIGHT_CANDIDATES}) OR pass --frames_csv.\n"
    #             f"If passing --frames_csv, it should include these identifier columns to merge on (subset ok): {present_ids}"
    #         )
    #     # Merge with frames CSV
    #     frames = pd.read_csv(args.frames_csv)
    #     # choose join keys as intersection of ID_CANDIDATES present in both
    #     lhs_keys = [c for c in ID_CANDIDATES if c in df.columns]
    #     rhs_keys = [c for c in ID_CANDIDATES if c in frames.columns]
    #     join_keys = [c for c in lhs_keys if c in rhs_keys]
    #     if not join_keys:
    #         raise SystemExit(
    #             "Cannot merge recall and frames CSVs: no common identifier columns found.\n"
    #             f"Recall CSV has: {lhs_keys}\nFrames CSV has: {rhs_keys}\n"
    #             f"Ensure they share at least one of: {ID_CANDIDATES}"
    #         )
    #     weight_col = find_first(list(frames.columns), WEIGHT_CANDIDATES)
    #     if weight_col is None:
    #         raise SystemExit(
    #             f"Frames CSV lacks a frames/weight column. Expected one of: {WEIGHT_CANDIDATES}\n"
    #             f"Frames CSV columns: {list(frames.columns)}"
    #         )
    #     df = df.merge(frames[join_keys + [weight_col]], on=join_keys, how="left")

    # validate weights
    # if df[weight_col].isna().any():
    #     missing = df[df[weight_col].isna()]
    #     print(f"[warn] {missing.shape[0]} rows have missing weights after merge and will be dropped.")
    df[weight_col] = pd.to_numeric(df[weight_col].astype(str).str.split('/').str[0], errors="coerce")
    df = df[df[weight_col].fillna(0) > 0].copy()

    # group by model and compute weighted means per recall column
    grouped = df.groupby(model_col, dropna=False)
    rows = []
    for model_name, g in grouped:
        row = {model_col: model_name, '_total_frames': g[weight_col].sum()}
        for rc in recall_cols:
            row[f"weighted_{rc}"] = weighted_mean(g[rc], g[weight_col])
        rows.append(row)
    out = pd.DataFrame(rows).set_index(model_col).sort_index()
    out.to_csv(args.out_csv, index=True)
    print(f"Saved weighted means to: {args.out_csv}")
    print(out)

if __name__ == "__main__":
    main()
