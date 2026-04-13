#!/usr/bin/env python3

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_dataset_meta(dataset_name: str) -> tuple[str, str]:
    name = str(dataset_name).lower()
    if "file" in name:
        category = "File"
    elif "registry" in name:
        category = "Registry"
    elif "network" in name:
        category = "Network"
    else:
        category = "Unknown"

    m = re.search(r"strategy_([a-z0-9]+)", name)
    strategy = m.group(1).upper() if m else "Unknown"
    return category, strategy


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "result" / "embedding_model_benchmark_quick" / "summary.csv"

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "category" not in df.columns or "strategy" not in df.columns:
        df[["category", "strategy"]] = df["dataset"].apply(
            lambda d: pd.Series(_parse_dataset_meta(d))
        )
    if "preprocess" not in df.columns:
        df["preprocess"] = "raw"

    df["model_variant"] = df["model"].astype(str) + " [" + df["preprocess"].astype(str) + "]"

    # Rank each method*preprocess variant inside each dataset, then draw top-10 bars
    ranked = (
        df.groupby(["dataset", "category", "strategy", "model_variant"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), n_rows=("n_rows", "sum"))
        .sort_values(["dataset", "accuracy"], ascending=[True, False])
    )

    print("Top-10 model*preprocess per dataset:")
    print(
        ranked.groupby("dataset", group_keys=False)
        .head(10)
        [["dataset", "model_variant", "accuracy", "n_rows"]]
        .to_string(index=False)
    )

    out_dir = repo_root / "result" / "embedding_model_benchmark_quick" / "dataset_top10"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, g in ranked.groupby("dataset", sort=True):
        top = g.head(10).copy()
        if top.empty:
            continue

        # Show highest at top
        top = top.sort_values("accuracy", ascending=True)

        fig_h = max(4.5, 0.55 * len(top) + 1.8)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        bars = ax.barh(top["model_variant"], top["accuracy"], color="#4C78A8")

        for b, val in zip(bars, top["accuracy"]):
            ax.text(
                min(1.02, float(val) + 0.01),
                b.get_y() + b.get_height() / 2,
                f"{float(val):.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )

        category = top["category"].iloc[0] if "category" in top.columns else "Unknown"
        strategy = top["strategy"].iloc[0] if "strategy" in top.columns else "Unknown"
        ax.set_title(f"Top 10 model*preprocess - {dataset_name} ({category} {strategy})", fontsize=12)
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Model [preprocess]")
        ax.set_xlim(0, 1.08)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.axvline(0.5, color="red", linestyle=":", alpha=0.5)
        plt.tight_layout()

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(dataset_name)).strip("_")
        out_file = out_dir / f"top10__{safe_name}.png"
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
