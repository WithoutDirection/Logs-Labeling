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
    csv_path = repo_root / "result" / "embedding_model_benchmark" / "summary.csv"

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

    preferred_categories = ["File", "Registry", "Network"]
    categories = [c for c in preferred_categories if c in set(df["category"])]
    categories += [c for c in sorted(df["category"].unique()) if c not in categories]

    strategy_order = sorted(df["strategy"].unique(), key=lambda s: (len(s), s))

    model_order = (
        df.groupby("model_variant", as_index=False)["accuracy"]
        .mean()
        .sort_values("accuracy", ascending=False)["model_variant"]
        .tolist()
    )

    display_df = (
        df[["model_variant", "category", "strategy", "accuracy", "n_rows"]]
        .sort_values(["category", "strategy", "accuracy"], ascending=[True, True, False])
    )
    print("Per-dataset Accuracy (not combined):")
    print(display_df.to_string(index=False))

    n_rows = min(3, max(1, len(categories)))
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(max(14, len(strategy_order) * 1.4), 5 * n_rows),
        sharex=False,
    )
    if n_rows == 1:
        axes = [axes]

    for idx, category in enumerate(categories[:n_rows]):
        ax = axes[idx]
        sub = df[df["category"] == category].copy()

        piv = (
            sub.pivot_table(index="strategy", columns="model_variant", values="accuracy", aggfunc="mean")
            .reindex(index=[s for s in strategy_order if s in set(sub["strategy"])])
        )

        model_cols = [m for m in model_order if m in piv.columns]
        piv = piv[model_cols]

        piv.plot(kind="bar", ax=ax, width=0.85)
        ax.set_title(f"{category} Accuracy by Strategy - dataset-first non-combined", fontsize=13)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Strategy")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.5)
        ax.legend(title="Model", ncol=min(4, max(1, len(model_cols))), loc="upper right")
        ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()

    out_file = repo_root / "result" / "embedding_model_benchmark" / "accuracy_dataset_first_rows.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print(f"\nVisualization saved to {out_file}")


if __name__ == "__main__":
    main()
