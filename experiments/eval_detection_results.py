#!/usr/bin/env python3
"""
Evaluate Stage II anomaly detection directly from Detection_Results folders.

Expected inputs:
  - data/Detection_Results/{dataset_id}_detection/
  - data/groundtruth/input_logs_labeled/{dataset_id}.csv

Ground-truth convention:
    - Technique == False => benign (0)
    - Technique has any other value => anomaly (1)

This bypasses Stage III and evaluates anomaly outputs directly against the
ground-truth Technique column.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _iter_detection_dirs(input_dir: Path) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob("*_detection") if p.is_dir())


def _candidate_dataset_ids(dataset_name: str) -> list[str]:
    candidates = [dataset_name]
    for suffix in ("_raw_events", "_events", "_raw"):
        if dataset_name.endswith(suffix):
            candidates.append(dataset_name[: -len(suffix)])
            break
    return candidates


def _extract_original_idx_from_logid(logid_series: pd.Series) -> pd.Series:
    """Extract original row index from LogID values ending with _{idx}."""
    return pd.to_numeric(
        logid_series.astype(str).str.extract(r"_(\d+)$", expand=False),
        errors="coerce",
    )


def _align_truth_from_groundtruth(det_df: pd.DataFrame, gt_df: pd.DataFrame, dataset_name: str) -> tuple[pd.Series, int]:
    """Align detection rows to ground truth.

    Preference order:
    1. Use LogID-derived original index when available.
    2. Fallback to strict length-based alignment.
    """
    if "Technique" not in gt_df.columns:
        raise ValueError(f"{dataset_name}: ground truth missing Technique column")

    gt_truth = _technique_to_anomaly(gt_df["Technique"]).reset_index(drop=True)

    if "LogID" in det_df.columns:
        idx = _extract_original_idx_from_logid(det_df["LogID"])
        valid = idx.notna() & idx.ge(0) & idx.lt(len(gt_truth))

        aligned = pd.Series(np.nan, index=det_df.index, dtype=float)
        if valid.any():
            aligned.loc[valid] = gt_truth.iloc[idx.loc[valid].astype(int)].to_numpy(dtype=float)

        return aligned, int((~valid).sum())

    if len(det_df) != len(gt_truth):
        raise ValueError(
            f"{dataset_name}: length mismatch and no LogID for alignment "
            f"(detection={len(det_df)} groundtruth={len(gt_truth)})"
        )

    return pd.Series(gt_truth.to_numpy(dtype=float), index=det_df.index), 0


def _technique_to_anomaly(series: pd.Series) -> pd.Series:
    """Map Technique values to anomaly labels.

    False/empty-like values are benign (0). Any other value is anomaly (1).
    """
    values = series.fillna("").astype(str).str.strip()
    lower = values.str.lower()
    benign_tokens = {"", "false", "0", "none", "nan", "null", "[]", "{}"}
    return (~lower.isin(benign_tokens)).astype(int)


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
    threshold_df.to_csv(out_dir / f"{score_col}_threshold_sweep.csv", index=False)

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


def _load_detection_dataset(path: Path) -> pd.DataFrame:
    ds = Dataset.load_from_disk(str(path))
    keep_cols = [
        "ensemble_score",
        "ensemble_raw_score",
        "ensemble_label",
        "ensemble_raw_label",
        "LogID",
    ]
    cols = [c for c in keep_cols if c in ds.column_names]
    return pd.DataFrame({c: ds[c] for c in cols})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage II anomaly detection directly from Detection_Results folders.")
    parser.add_argument(
        "--input-dir",
        default="/tmp2/b11902050/Logs-Labeling/data/Detection_Results",
        help="Directory containing *_detection folders.",
    )
    parser.add_argument(
        "--groundtruth-dir",
        default="/tmp2/b11902050/Logs-Labeling/data/groundtruth/input_logs_labeled",
        help="Directory containing ground-truth labeled CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp2/b11902050/Logs-Labeling/result/anomaly_eval_from_detection",
        help="Directory for evaluation outputs.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    groundtruth_dir = Path(args.groundtruth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for det_dir in _iter_detection_dirs(input_dir):
        dataset_name = det_dir.name.replace("_detection", "")
        gt_csv = None
        for candidate in _candidate_dataset_ids(dataset_name):
            candidate_path = groundtruth_dir / f"{candidate}.csv"
            if candidate_path.exists():
                gt_csv = candidate_path
                break

        if gt_csv is None:
            continue

        det_df = _load_detection_dataset(det_dir)
        gt_df = pd.read_csv(gt_csv)

        try:
            aligned_truth, dropped = _align_truth_from_groundtruth(det_df, gt_df, dataset_name)
        except ValueError as exc:
            print(f"[Skip] {exc}")
            continue

        slim = pd.DataFrame(index=det_df.index)
        slim["dataset"] = dataset_name

        if "ensemble_score" in det_df.columns:
            slim["anomaly_score"] = pd.to_numeric(det_df["ensemble_score"], errors="coerce")
        if "ensemble_raw_score" in det_df.columns:
            slim["anomaly_raw_score"] = pd.to_numeric(det_df["ensemble_raw_score"], errors="coerce")
        if "ensemble_label" in det_df.columns:
            slim["anomaly_label"] = pd.to_numeric(det_df["ensemble_label"], errors="coerce")
        if "ensemble_raw_label" in det_df.columns:
            slim["anomaly_raw_label"] = pd.to_numeric(det_df["ensemble_raw_label"], errors="coerce")

        slim["y_true"] = aligned_truth
        if dropped:
            print(f"[Info] {dataset_name}: dropped {dropped} rows due to invalid LogID indices")

        slim = slim.dropna(subset=["y_true"]).copy()
        if slim.empty:
            print(f"[Skip] no aligned rows left for {dataset_name}")
            continue

        slim["y_true"] = slim["y_true"].astype(int)
        all_rows.append(slim)

    if not all_rows:
        raise FileNotFoundError(f"No usable detection results found in {input_dir}")

    merged = pd.concat(all_rows, ignore_index=True)
    merged.to_csv(output_dir / "merged_for_eval.csv", index=False)

    summaries = []
    if "anomaly_score" in merged.columns:
        summaries.append(evaluate_score(merged, "anomaly_score", output_dir))
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
