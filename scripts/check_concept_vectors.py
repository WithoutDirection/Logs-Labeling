import os
import sys
import argparse
import numpy as np
import pyarrow.feather as feather

# Allow importing project config when running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # noqa: E402


def load_vectors(arrow_path: str) -> np.ndarray:
    table = feather.read_table(arrow_path)
    if "concept_vector" in table.column_names:
        return np.array(table["concept_vector"].to_pylist())
    return table.to_pandas().values


def check_arrow_file(arrow_path: str) -> dict:
    vectors = load_vectors(arrow_path)
    nan_mask = np.isnan(vectors)
    inf_mask = ~np.isfinite(vectors)
    return {
        "shape": vectors.shape,
        "nan_count": int(nan_mask.sum()),
        "inf_count": int(inf_mask.sum()),
        "all_finite": bool(np.isfinite(vectors).all()),
    }


def main():
    parser = argparse.ArgumentParser(description="Scan ConceptVectors for NaN/Inf values")
    parser.add_argument(
        "--root",
        default=config.CONCEPT_VECTORS_DIR,
        help="Root directory of ConceptVectors",
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.exists(root):
        print(f"ConceptVectors root not found: {root}")
        return

    issues = []
    total_files = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".arrow"):
                continue
            total_files += 1
            arrow_path = os.path.join(dirpath, name)
            try:
                result = check_arrow_file(arrow_path)
                if not result["all_finite"] or result["nan_count"] > 0 or result["inf_count"] > 0:
                    issues.append((arrow_path, result))
                print(f"OK {arrow_path}: shape={result['shape']} all_finite={result['all_finite']} nan={result['nan_count']} inf={result['inf_count']}")
            except Exception as e:
                issues.append((arrow_path, {"error": str(e)}))
                print(f"FAIL {arrow_path}: {e}")

    print("\nSummary")
    print("======")
    print(f"Scanned Arrow files: {total_files}")
    print(f"Files with issues: {len(issues)}")
    if issues:
        for path, info in issues:
            print(f"- {path}: {info}")


if __name__ == "__main__":
    main()
