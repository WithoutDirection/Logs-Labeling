#!/usr/bin/env python3
"""
Append MITRE ATT&CK External IDs (T-codes) to a reference CSV.

Takes an existing MITRE techniques CSV (e.g. combined.csv or MitreTechniquesTokens_V5.csv)
that has a ``technique`` name column but is missing the ``external_id`` (Txxxx) column,
fetches the authoritative name → ID mapping from the live MITRE STIX data via
``mitreattack-python``, and writes a new CSV with the ``external_id`` column appended.

Resolution order for each technique name:
  1. Manual map CSV  (``--manual_map_csv``)  — for overrides or names not in STIX data
  2. Live STIX data  (``mitreattack.attackToExcel``)

If any technique names cannot be resolved, the script saves a ``missing_external_ids.csv``
alongside the output file, prints the unresolved names, and exits with code 1 so the caller
can fill in the gaps and rerun with ``--manual_map_csv``.

Usage:
    python append_mitre_external_ids.py \\
        --mitre_csv  data/reference_resources/combined.csv \\
        --output_csv data/reference_resources/combined_with_ids.csv

    # with manual overrides:
    python append_mitre_external_ids.py \\
        --mitre_csv  data/reference_resources/combined.csv \\
        --output_csv data/reference_resources/combined_with_ids.csv \\
        --manual_map_csv manual_id_map.csv

Dependencies:
    pip install mitreattack-python
"""

import argparse
import os
from typing import Dict

import pandas as pd

try:
    import mitreattack.attackToExcel.attackToExcel as attackToExcel
    import mitreattack.attackToExcel.stixToDf as stixToDf
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "mitreattack package is required. Install it via `pip install mitreattack-python`."
    ) from exc


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _load_manual_map(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "technique" not in df.columns or "external_id" not in df.columns:
        raise ValueError("Manual map CSV must include columns: technique, external_id")
    return {
        _normalize_name(r["technique"]): str(r["external_id"]).strip()
        for _, r in df.iterrows()
        if str(r.get("external_id", "")).strip()
    }


def _build_name_to_external_id(domain: str) -> Dict[str, str]:
    attackdata = attackToExcel.get_stix_data(domain)
    techniques_df = stixToDf.techniquesToDf(attackdata, domain)["techniques"].reset_index(drop=True)
    if "name" not in techniques_df.columns or "ID" not in techniques_df.columns:
        raise ValueError("mitreattack techniques DataFrame missing 'name' or 'ID'")

    mapping: Dict[str, str] = {}
    for _, row in techniques_df.iterrows():
        name = _normalize_name(row["name"])
        external_id = str(row["ID"]).strip()
        if name and external_id:
            mapping[name] = external_id
    return mapping


def append_external_ids(
    mitre_csv: str,
    output_csv: str,
    manual_map_csv: str | None = None,
    domain: str = "enterprise-attack",
) -> None:
    if not os.path.isfile(mitre_csv):
        raise FileNotFoundError(f"MITRE CSV not found: {mitre_csv}")

    mitre_df = pd.read_csv(mitre_csv)
    if "technique" not in mitre_df.columns:
        raise ValueError("Expected 'technique' column in MITRE CSV")

    name_to_id = _build_name_to_external_id(domain)
    manual_map = _load_manual_map(manual_map_csv) if manual_map_csv else {}

    def resolve_external_id(name: str) -> str:
        key = _normalize_name(name)
        return manual_map.get(key) or name_to_id.get(key, "")

    mitre_df["external_id"] = mitre_df["technique"].astype(str).apply(resolve_external_id)

    missing = mitre_df[mitre_df["external_id"] == ""]["technique"].dropna().astype(str).tolist()
    if missing:
        missing_path = os.path.join(os.path.dirname(output_csv), "missing_external_ids.csv")
        pd.DataFrame({"technique": missing, "external_id": ""}).to_csv(missing_path, index=False)
        print("Missing external IDs for techniques. Please fill them and rerun:")
        for name in missing:
            print(f"  - {name}")
        print(f"Missing list saved to: {missing_path}")
        raise SystemExit(1)

    mitre_df.to_csv(output_csv, index=False)
    print(f"Wrote CSV with external_id to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mitre_csv", required=True, help="Path to MitreTechniquesTokens_V5.csv")
    parser.add_argument("--output_csv", required=True, help="Output CSV path with external_id column")
    parser.add_argument("--manual_map_csv", default=None, help="Optional manual mapping CSV: technique, external_id")
    parser.add_argument("--domain", default="enterprise-attack")
    args = parser.parse_args()

    append_external_ids(
        mitre_csv=args.mitre_csv,
        output_csv=args.output_csv,
        manual_map_csv=args.manual_map_csv,
        domain=args.domain,
    )
