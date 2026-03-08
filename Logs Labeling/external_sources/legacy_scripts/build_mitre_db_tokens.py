import os
from typing import Dict, List

import pandas as pd

from text_processor import TextProcessor
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import mitreattack.attackToExcel.stixToDf as stixToDf
    import mitreattack.attackToExcel.attackToExcel as attackToExcel
except ImportError as exc:  # pragma: no cover - environment-specific
    raise ImportError(
        "mitreattack package is required to build MITRE DB-enriched CSVs. "
        "Install it via `pip install mitreattack-python`.") from exc


def load_mitreattack_techniques(domain: str = "enterprise-attack") -> pd.DataFrame:
    """Load MITRE ATT&CK techniques via the mitreattack package.

    Returns the `techniques` DataFrame for the given domain.
    """
    attackdata = attackToExcel.get_stix_data(domain)
    techniques_data = stixToDf.techniquesToDf(attackdata, domain)
    techniques_df = techniques_data["techniques"].reset_index(drop=True)
    return techniques_df


def build_db_text_mapping(domain: str = "enterprise-attack") -> Dict[str, str]:
    """Build a mapping from technique NAME to concatenated database text.

    This collects relevant textual fields from the mitreattack techniques
    DataFrame (excluding obvious non-text ID fields) and concatenates them
    into a single string per technique. The key is the *technique name*,
    which will be matched against the `technique` column in the local
    MITRE CSV. This avoids mismatch between ATT&CK external IDs (Txxxx)
    and local STIX-style `technique_id` values.
    """
    df = load_mitreattack_techniques(domain=domain)

    if "name" not in df.columns:
        raise ValueError("Expected column 'name' in mitreattack techniques DataFrame")

    # Choose object-type columns that are likely descriptive text, excluding
    # obvious ID fields.
    print(f"[Debug] Columns in STIX data: {df.columns.tolist()}")

    # Priority columns that contain the rich context we want. We match
    # both STIX-style (x_mitre_*) and attackToExcel-style names.
    preferred_names = {
        "name",
        "description",
        "detection",
        "platforms",
        "data sources",
        "data_sources",
        "tactics",
        "sub-technique of",
        "contributors",
        "supports remote",
        "impact type",
        "relationship citations",
        "x_mitre_detection",
        "x_mitre_platforms",
        "x_mitre_data_sources",
        "x_mitre_permissions_required",
        "x_mitre_system_requirements",
        "x_mitre_remote_support",
        "x_mitre_network_requirements",
        "permissions required",
        "system requirements",
        "network requirements",
    }

    def _norm_col(col_name: str) -> str:
        return col_name.strip().lower().replace("_", " ")

    # Filter for columns that actually exist in the dataframe (by normalized name)
    text_cols = []
    for c in df.columns:
        if _norm_col(c) in preferred_names:
            text_cols.append(c)

    # Add any other object columns that are not IDs and not already included
    for c in df.columns:
        if c in text_cols:
            continue
        if c in {"ID", "id", "stix_id", "type"}:
            continue
        if c.endswith("_id") or c.endswith("_ref"):
            continue
        if df[c].dtype == object:
            text_cols.append(c)

    print(f"[Debug] Selected text columns: {text_cols}")

    def row_to_text(row: pd.Series) -> str:
        parts: List[str] = []
        for col in text_cols:
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if isinstance(val, str):
                text = val.strip()
            else:
                text = str(val).strip()
            if text:
                # Prefix with column name to give some structure
                if col == "name":
                    parts.append(text)
                else:
                    parts.append(f"{col}: {text}")
        return "\n\n".join(parts)

    db_text_series = df.apply(row_to_text, axis=1)
    mapping: Dict[str, str] = {}
    for name, text in zip(df["name"].astype(str).tolist(), db_text_series.tolist()):
        mapping[name] = text

    return mapping


def build_mitre_techniques_with_db(
    mitre_csv: str,
    output_csv: str,
    domain: str = "enterprise-attack",
    bert_model_name: str | None = None,
) -> None:
    """Build a technique-level MITRE CSV enriched with mitreattack DB fields.

    - Reads the existing MITRE techniques tokens CSV (e.g.,
      `MitreTechniquesTokens_V5.csv`).
    - Uses the `mitreattack` Python package to fetch the official ATT&CK
      techniques database for the given domain.
    - For each technique_id, concatenates relevant database fields into
      an additional text block and appends it to the existing MITRE
      description text (no website/crawler code involved).
    - Re-runs the ConceptUML preprocessing (`TextProcessor`) so that
      `description`, `description_raw`, `all_text`, `tokens`, and
      `cleaned_tokens` all reflect the combined docs+database text.
    - Writes a new CSV at `output_csv` that can be used as an alternate
      MITRE source (e.g., in `test_caldera_logs.py`).
    """

    if not os.path.isfile(mitre_csv):
        raise FileNotFoundError(f"MITRE techniques CSV not found: {mitre_csv}")

    mitre_df = pd.read_csv(mitre_csv)

    if "technique" not in mitre_df.columns:
        raise ValueError("Expected 'technique' column in MITRE CSV for name-based join")

    db_text_map = build_db_text_mapping(domain=domain)

    combined_df = mitre_df.copy()

    # Attach database text by matching mitreattack `name` to local
    # MITRE `technique` name. This is more robust given that the local
    # `technique_id` uses STIX-style IDs while mitreattack uses external
    # IDs (e.g., T1112).
    combined_df["db_text"] = combined_df["technique"].map(
        lambda name: db_text_map.get(str(name), "")
    )

    # Build combined description = original description_raw + database text
    def _combine_row(row: pd.Series) -> str:
        base = str(row.get("description_raw", "")) if not pd.isna(row.get("description_raw", "")) else ""
        db_text = str(row.get("db_text", "")) if not pd.isna(row.get("db_text", "")) else ""
        if db_text.strip():
            if base.strip():
                return base + "\n\n" + db_text
            return db_text
        return base

    combined_texts = combined_df.apply(_combine_row, axis=1)

    # Set primary text fields to the combined docs+database text.
    combined_df["description"] = combined_texts
    combined_df["description_raw"] = combined_texts
    combined_df["all_text"] = combined_texts

    if bert_model_name is None:
        import config
        bert_model_name = getattr(config, "EXTERNAL_SOURCES_BERT_MODEL_NAME", getattr(config, "BERT_MODEL_NAME", "sentence-bert"))
    # Re-run ConceptUML preprocessing so tokens/cleaned_tokens reflect both
    # the original docs and the attached database text.
    tp = TextProcessor(bert_model_name=bert_model_name)
    tp.fit_zipf_filter(combined_df["description"].fillna("").astype(str).tolist())

    descriptions_clean: List[str] = []
    tokens_list: List[List[str]] = []
    cleaned_tokens_list: List[List[str]] = []

    for text in combined_df["description"].fillna("").astype(str).tolist():
        cleaned = tp.clean_text(text)
        descriptions_clean.append(cleaned)

        tokens = tp.tokenize(text, remove_stopwords=False, apply_zipf_filter=False)
        tokens_list.append(tokens)

        cleaned_tokens = tp.tokenize(text, remove_stopwords=True, apply_zipf_filter=True)
        cleaned_tokens_list.append(cleaned_tokens)

    combined_df["description_clean"] = descriptions_clean
    combined_df["tokens"] = tokens_list
    combined_df["cleaned_tokens"] = cleaned_tokens_list

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(
        f"Wrote MITRE techniques+DB CSV with {len(combined_df)} rows to {output_csv}"
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    
    

    # Original MITRE techniques tokens file
    root_data_dir = (
        getattr(config, "REFERENCE_RESOURCES_DIR", None) if config else None
    ) or os.path.normpath(os.path.join(base_dir, "..", "..", "data", "reference_resources"))

    mitre_original_csv = (
        getattr(config, "MITRE_TECHNIQUES_CSV", None) if config else None
    ) or os.path.join(root_data_dir, "MitreTechniquesTokens_V5.csv")

    # New output file enriched with mitreattack database fields
    mitre_with_db_csv = os.path.join(root_data_dir, "MitreTechniquesTokens_WithDB_V1.csv")

    build_mitre_techniques_with_db(
        mitre_csv=mitre_original_csv,
        output_csv=mitre_with_db_csv,
    )
