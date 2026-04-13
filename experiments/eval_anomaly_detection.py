#!/usr/bin/env python3
"""
Evaluate Stage II anomaly detection from labeled CSV outputs.

Expected columns in each *_Labeled.csv:
  - anomaly_score
Optional:
  - anomaly_raw_score
  - anomaly_label
  - anomaly_raw_label
    - original_idx

Ground-truth source:
    - data/groundtruth/input_logs_labeled/{dataset_id}.csv
    - Technique == False => benign (0)
    - Technique has any other value => anomaly (1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _iter_labeled_csvs(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*_Labeled.csv") if p.is_file())


def _dataset_id_from_labeled_path(csv_path: Path) -> str:
    name = csv_path.stem
    if name.endswith("_Labeled"):
        name = name[:-len("_Labeled")]
    for suffix in ("_raw_events", "_events", "_raw"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def _technique_to_anomaly(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    lower = values.str.lower()
    benign_tokens = {"", "false", "0", "none", "nan", "null", "[]", "{}"}
    return (~lower.isin(benign_tokens)).astype(int)


def _align_truth_from_groundtruth(df: pd.DataFrame, gt_df: pd.DataFrame, dataset_name: str) -> tuple[pd.Series, int]:
    if "Technique" not in gt_df.columns:
        raise ValueError(f"{dataset_name}: ground truth missing Technique column")

    gt_truth = _technique_to_anomaly(gt_df["Technique"]).reset_index(drop=True)

    if "original_idx" in df.columns:
        idx = pd.to_numeric(df["original_idx"], errors="coerce")
        valid = idx.notna() & idx.ge(0) & idx.lt(len(gt_truth))
        aligned = pd.Series(np.nan, index=df.index, dtype=float)
        if valid.any():
            aligned.loc[valid] = gt_truth.iloc[idx.loc[valid].astype(int)].to_numpy(dtype=float)
        return aligned, int((~valid).sum())

    if len(df) != len(gt_truth):
        raise ValueError(
            f"{dataset_name}: length mismatch and no original_idx for alignment "
            f"(labeled={len(df)}, groundtruth={len(gt_truth)})"
        )

    return pd.Series(gt_truth.to_numpy(dtype=float), index=df.index), 0


def _topk_metrics(y_true: np.ndarray, scores: np.ndarray, frac: float) -> dict[str, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(n * frac)))
    order = np.argsort(scores)[::-1][:k]
    hits = y_true[order].sum()
    total_pos = y_true.sum()
    return {
        "k": float(k),
        "precision": float(hits / k) if k else 0.0,
        "recall": float(hits / total_pos) if total_pos else 0.0,
    }


def _threshold_sweep(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    thresholds = np.unique(np.round(np.quantile(scores, np.linspace(0.0, 1.0, 501)), 6))
    rows = []
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fp_per_10k = fp / len(y_true) * 10000.0 if len(y_true) else 0.0
        rows.append({
            "threshold": float(t),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fp": fp,
            "tp": tp,
            "fn": fn,
            "fp_per_10k_events": float(fp_per_10k),
        })
    return pd.DataFrame(rows).sort_values("threshold", ascending=False)


def _dataset_hit_metrics(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    hit1 = 0
    hit5 = 0
    first_ranks = []
    valid = 0
    for _, g in df.groupby("dataset"):
        y = g["y_true"].to_numpy(dtype=int)
        if y.sum() == 0:
            continue
        valid += 1
        scores = g[score_col].to_numpy(dtype=float)
        order = np.argsort(scores)[::-1]
        ranked_truth = y[order]
        malicious_positions = np.where(ranked_truth == 1)[0]
        if len(malicious_positions) == 0:
            continue
        first_rank = int(malicious_positions[0]) + 1
        first_ranks.append(first_rank)
        if first_rank <= 1:
            hit1 += 1
        if first_rank <= 5:
            hit5 += 1
    return {
        "dataset_count": float(valid),
        "dataset_hit_at_1": float(hit1 / valid) if valid else 0.0,
        "dataset_hit_at_5": float(hit5 / valid) if valid else 0.0,
        "dataset_mean_first_malicious_rank": float(np.mean(first_ranks)) if first_ranks else np.nan,
        "dataset_median_first_malicious_rank": float(np.median(first_ranks)) if first_ranks else np.nan,
    }


def evaluate_score(df: pd.DataFrame, score_col: str, out_dir: Path) -> dict[str, float]:
    work_df = df.dropna(subset=[score_col]).copy()
    y_true = work_df["y_true"].to_numpy(dtype=int)
    scores = work_df[score_col].to_numpy(dtype=float)

    summary = {
        "score_col": score_col,
        "n_rows": float(len(work_df)),
        "n_positives": float(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "pr_auc": float(average_precision_score(y_true, scores)) if y_true.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
    }

    for frac, label in [(0.001, "top_0_1pct"), (0.005, "top_0_5pct"), (0.01, "top_1pct"), (0.05, "top_5pct")]:
        metrics = _topk_metrics(y_true, scores, frac)
        summary[f"{label}_k"] = metrics["k"]
        summary[f"{label}_precision"] = metrics["precision"]
        summary[f"{label}_recall"] = metrics["recall"]

    threshold_df = _threshold_sweep(y_true, scores)
    threshold_path = out_dir / f"{score_col}_threshold_sweep.csv"
    threshold_df.to_csv(threshold_path, index=False)

    if not threshold_df.empty:
        best_f1_row = threshold_df.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
        summary["best_f1_threshold"] = float(best_f1_row["threshold"])
        summary["best_f1"] = float(best_f1_row["f1"])
        summary["best_f1_precision"] = float(best_f1_row["precision"])
        summary["best_f1_recall"] = float(best_f1_row["recall"])

    summary.update(_dataset_hit_metrics(work_df, score_col))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold": np.append(thresholds, np.nan),
    })
    pr_df.to_csv(out_dir / f"{score_col}_pr_curve.csv", index=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage II anomaly detection using labeled CSV outputs.")
    parser.add_argument(
        "--input-dir",
        default="/tmp2/b11902050/Logs-Labeling/result/Labeling_Results",
        help="Directory containing *_Labeled.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp2/b11902050/Logs-Labeling/result/anomaly_eval",
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--groundtruth-dir",
        default="/tmp2/b11902050/Logs-Labeling/data/groundtruth/input_logs_labeled",
        help="Directory containing input_logs_labeled ground-truth CSVs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    groundtruth_dir = Path(args.groundtruth_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for csv_path in _iter_labeled_csvs(input_dir):
        df = pd.read_csv(csv_path)
        if "anomaly_score" not in df.columns:
            continue
        dataset_name = _dataset_id_from_labeled_path(csv_path)
        gt_csv = groundtruth_dir / f"{dataset_name}.csv"
        if not gt_csv.exists():
            print(f"[Skip] ground-truth CSV not found for {dataset_name}: {gt_csv}")
            continue

        gt_df = pd.read_csv(gt_csv)

        keep_cols = ["anomaly_score"]
        optional_cols = ["anomaly_raw_score", "anomaly_label", "anomaly_raw_label", "original_idx"]
        keep_cols.extend([c for c in optional_cols if c in df.columns])
        slim = df[keep_cols].copy()
        slim["dataset"] = dataset_name

        try:
            aligned_truth, dropped = _align_truth_from_groundtruth(slim, gt_df, dataset_name)
        except ValueError as exc:
            print(f"[Skip] {exc}")
            continue

        slim["y_true"] = aligned_truth
        if dropped:
            print(f"[Info] {dataset_name}: dropped {dropped} rows due to invalid original_idx")

        slim = slim.dropna(subset=["y_true"]).copy()
        if slim.empty:
            print(f"[Skip] no aligned rows left for {dataset_name}")
            continue

        slim["y_true"] = slim["y_true"].astype(int)
        all_rows.append(slim)

    if not all_rows:
        raise FileNotFoundError(f"No usable labeled CSVs found in {input_dir}")

    merged = pd.concat(all_rows, ignore_index=True)
    merged.to_csv(output_dir / "merged_for_eval.csv", index=False)

    summaries = [evaluate_score(merged, "anomaly_score", output_dir)]
    if "anomaly_raw_score" in merged.columns:
        summaries.append(evaluate_score(merged, "anomaly_raw_score", output_dir))

    if "anomaly_label" in merged.columns:
        y_true = merged["y_true"].to_numpy(dtype=int)
        y_pred = merged["anomaly_label"].fillna(0).to_numpy(dtype=int)
        summaries.append({
            "score_col": "anomaly_label",
            "n_rows": float(len(merged)),
            "n_positives": float(y_true.sum()),
            "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
