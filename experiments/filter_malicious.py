#!/usr/bin/env python3
"""
Filter labeled CSV outputs and keep only rows that satisfy the intended
"malicious" thresholds.

By default this applies the project thresholds from config.py:
  - predicted_technique_1_threat_confidence >= LABELING_CONFIDENCE_THRESHOLD
  - predicted_technique_1_similarity >= LABELING_SIMILARITY_THRESHOLD

Filtered CSVs are written to:
  result/Labeling_Results_Malicious/
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PKG_DIR = REPO_ROOT / "Logs Labeling"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

_config_spec = importlib.util.spec_from_file_location("project_config", PKG_DIR / "config.py")
if _config_spec is None or _config_spec.loader is None:
    raise ImportError(f"Cannot load config.py from {PKG_DIR}")
config = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(config)


DEFAULT_INPUT_DIR = Path(config.LABELING_RESULTS_DIR)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "result" / "Labeling_Results_Malicious"


def _iter_labeled_csvs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.glob("*_Labeled.csv")
        if p.is_file()
    )


def filter_single_file(
    csv_path: Path,
    output_dir: Path,
    confidence_threshold: float,
    similarity_threshold: float,
    anomaly_threshold: float | None,
) -> dict[str, object]:
    df = pd.read_csv(csv_path)

    required_cols = {
        "predicted_technique_1_name",
        "predicted_technique_1_threat_confidence",
        "predicted_technique_1_similarity",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path.name} is missing required columns: {missing}")

    name_col = df["predicted_technique_1_name"].fillna("").astype(str).str.strip()
    conf_col = pd.to_numeric(df["predicted_technique_1_threat_confidence"], errors="coerce")
    sim_col = pd.to_numeric(df["predicted_technique_1_similarity"], errors="coerce")

    mask = (
        name_col.ne("")
        & ~name_col.isin(["TBD", "Benign", "Unknown"])
        & conf_col.ge(confidence_threshold)
        & sim_col.ge(similarity_threshold)
    )

    if anomaly_threshold is not None:
        if "anomaly_score" not in df.columns:
            raise ValueError(
                f"{csv_path.name} does not contain anomaly_score, "
                "but --anomaly-threshold was provided."
            )
        anomaly_col = pd.to_numeric(df["anomaly_score"], errors="coerce")
        mask &= anomaly_col.ge(anomaly_threshold)

    filtered_df = df[mask].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / csv_path.name
    filtered_df.to_csv(output_path, index=False)

    return {
        "dataset": csv_path.stem.replace("_Labeled", ""),
        "input_rows": int(len(df)),
        "kept_rows": int(len(filtered_df)),
        "kept_ratio": float(len(filtered_df) / len(df)) if len(df) else 0.0,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter labeled CSVs and keep only malicious rows.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_DIR),
        help="Input labeled CSV file or directory. Defaults to result/Labeling_Results/",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for filtered CSV outputs.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(config.LABELING_CONFIDENCE_THRESHOLD),
        help="Minimum predicted_technique_1_threat_confidence to keep.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=float(config.LABELING_SIMILARITY_THRESHOLD),
        help="Minimum predicted_technique_1_similarity to keep.",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=None,
        help="Optional minimum anomaly_score to keep.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    csv_paths = _iter_labeled_csvs(input_path)
    if not csv_paths:
        raise FileNotFoundError(f"No labeled CSVs found under: {input_path}")

    rows = []
    for csv_path in csv_paths:
        result = filter_single_file(
            csv_path=csv_path,
            output_dir=output_dir,
            confidence_threshold=args.confidence_threshold,
            similarity_threshold=args.similarity_threshold,
            anomaly_threshold=args.anomaly_threshold,
        )
        rows.append(result)
        print(
            f"{csv_path.name}: kept {result['kept_rows']}/{result['input_rows']} "
            f"({result['kept_ratio']:.1%})"
        )

    summary_df = pd.DataFrame(rows).sort_values(["kept_ratio", "dataset"], ascending=[False, True])
    summary_path = output_dir / "filter_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved filtered CSVs to: {output_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
