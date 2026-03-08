import os
import json
from typing import List, Dict, Any

import pandas as pd

from text_processor import TextProcessor


def load_cti_code_json(filepath: str) -> Dict[str, Any]:
    """Load a single CTI code-only JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_code_rows(cti_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a CTI JSON object into per-code-snippet rows.

    Each row corresponds to one code_content under a procedure_example.
    """
    rows: List[Dict[str, Any]] = []

    tactic_id = cti_obj.get("tactic_id")
    tactic_name = cti_obj.get("tactic_name")

    for tech in cti_obj.get("techniques", []):
        technique_id = tech.get("technique_id")
        technique_name = tech.get("technique_name")

        for proc in tech.get("procedure_examples", []):
            procedure_id = proc.get("procedure_id")
            procedure_name = proc.get("name")  # e.g., group/campaign/tool name
            procedure_desc = proc.get("description", "")

            code_snippets = proc.get("code_snippets", [])
            if not code_snippets:
                # Still create a row so that procedures without explicit code
                # snippets can be represented via their description.
                rows.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "technique_id": technique_id,
                        "technique_name": technique_name,
                        "procedure_id": procedure_id,
                        "procedure_name": procedure_name,
                        "source_url": None,
                        "code_content": None,
                        "description": procedure_desc,
                    }
                )
                continue

            for snippet in code_snippets:
                url = snippet.get("url")
                code_content = snippet.get("code_content", "")

                # Use a combined description field as the main text for
                # preprocessing / embeddings.
                # This keeps both the natural-language description and the
                # code snippet together.
                combined_text_parts: List[str] = []
                if isinstance(procedure_desc, str) and procedure_desc.strip():
                    combined_text_parts.append(procedure_desc)
                if isinstance(code_content, str) and code_content.strip():
                    combined_text_parts.append(code_content)
                combined_text = "\n\n".join(combined_text_parts)

                rows.append(
                    {
                        "tactic_id": tactic_id,
                        "tactic_name": tactic_name,
                        "technique_id": technique_id,
                        "technique_name": technique_name,
                        "procedure_id": procedure_id,
                        "procedure_name": procedure_name,
                        "source_url": url,
                        "code_content": code_content,
                        "description": combined_text,
                    }
                )

    return rows


def build_mitre_code_tokens(
    cti_dir: str,
    output_csv: str,
    bert_model_name: str | None = None,
) -> None:
    """Build a preprocessed CSV of MITRE code snippets for use as a source.

    - Reads all JSON files under ``cti_dir`` (e.g., data/cti_code_only).
    - Flattens them into per-code-snippet rows.
    - Applies the same preprocessing as other external sources via
      ``TextProcessor`` to generate ``tokens`` and ``cleaned_tokens``.
    - Writes a CSV at ``output_csv`` suitable for ``ExternalSourceManager``.
    """

    if not os.path.isdir(cti_dir):
        raise FileNotFoundError(f"CTI directory not found: {cti_dir}")

    all_rows: List[Dict[str, Any]] = []

    for filename in os.listdir(cti_dir):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(cti_dir, filename)
        try:
            cti_obj = load_cti_code_json(path)
            rows = collect_code_rows(cti_obj)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")

    if not all_rows:
        print("No rows collected from CTI code JSONs; nothing to write.")
        return

    df = pd.DataFrame(all_rows)

    # Ensure there is a description column for preprocessing
    if "description" not in df.columns:
        df["description"] = ""

    if bert_model_name is None:
        try:
            import config

            bert_model_name = getattr(config, "EXTERNAL_SOURCES_BERT_MODEL_NAME", getattr(config, "BERT_MODEL_NAME", "sentence-bert"))
        except Exception:
            bert_model_name = "sentence-bert"

    # Use the same preprocessing pipeline as other external sources.
    tp = TextProcessor(bert_model_name=bert_model_name)

    # Fit Zipf filter on all descriptions
    tp.fit_zipf_filter(df["description"].fillna("").astype(str).tolist())

    descriptions_clean: List[str] = []
    tokens_list: List[List[str]] = []
    cleaned_tokens_list: List[List[str]] = []

    for text in df["description"].fillna("").astype(str).tolist():
        cleaned = tp.clean_text(text)
        descriptions_clean.append(cleaned)

        tokens = tp.tokenize(text, remove_stopwords=False, apply_zipf_filter=False)
        tokens_list.append(tokens)

        cleaned_tokens = tp.tokenize(text, remove_stopwords=True, apply_zipf_filter=True)
        cleaned_tokens_list.append(cleaned_tokens)

    df["description_clean"] = descriptions_clean
    df["tokens"] = tokens_list
    df["cleaned_tokens"] = cleaned_tokens_list

    # Sort for reproducibility (by technique and procedure if available)
    sort_cols = [col for col in ["technique_id", "procedure_id", "source_url"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote MITRE code tokens CSV with {len(df)} rows to {output_csv}")


def build_mitre_techniques_with_code(
    mitre_csv: str,
    code_tokens_csv: str,
    output_csv: str,
    bert_model_name: str | None = None,
) -> None:
    """Build a technique-level MITRE CSV where docs are augmented with code.

    This treats code snippets as *additional text* for each technique, grouped
    by ``technique_name`` from the CTI JSON and merged into the MITRE
    ``technique`` column. The resulting CSV can be used as a drop-in
    replacement for the original MITRE techniques file for indexing and
    concept extraction, without modifying the original.
    """

    if not os.path.isfile(mitre_csv):
        raise FileNotFoundError(f"MITRE techniques CSV not found: {mitre_csv}")
    if not os.path.isfile(code_tokens_csv):
        raise FileNotFoundError(f"MITRE code-tokens CSV not found: {code_tokens_csv}")

    mitre_df = pd.read_csv(mitre_csv)
    code_df = pd.read_csv(code_tokens_csv)

    if "technique_name" not in code_df.columns:
        raise ValueError("Expected 'technique_name' column in code tokens CSV")

    # Aggregate all code+procedure text per technique name
    def _agg_descriptions(series: pd.Series) -> str:
        parts = [str(x) for x in series.dropna().tolist() if str(x).strip()]
        return "\n\n".join(parts)

    code_by_technique_name: Dict[str, str] = (
        code_df.groupby("technique_name")["description"].apply(_agg_descriptions).to_dict()
    )

    combined_df = mitre_df.copy()

    # Attach code text by matching CTI "technique_name" to MITRE "technique".
    combined_df["code_text"] = combined_df["technique"].map(
        lambda name: code_by_technique_name.get(str(name), "")
    )

    # Build combined description = original description_raw + code_text
    def _combine_row(row: pd.Series) -> str:
        base = str(row.get("description_raw", "")) if not pd.isna(row.get("description_raw", "")) else ""
        code = str(row.get("code_text", "")) if not pd.isna(row.get("code_text", "")) else ""
        if code.strip():
            if base.strip():
                return base + "\n\n" + code
            return code
        return base

    combined_texts = combined_df.apply(_combine_row, axis=1)

    # For downstream code that auto-selects description columns, we set all
    # primary text fields to the combined doc+code text.
    combined_df["description"] = combined_texts
    combined_df["description_raw"] = combined_texts
    combined_df["all_text"] = combined_texts

    # Re-run the ConceptUML preprocessing pipeline over the combined text so
    # tokens/cleaned_tokens reflect both docs and code.
    if bert_model_name is None:
        try:
            import config

            bert_model_name = getattr(config, "EXTERNAL_SOURCES_BERT_MODEL_NAME", getattr(config, "BERT_MODEL_NAME", "sentence-bert"))
        except Exception:
            bert_model_name = "sentence-bert"

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
        f"Wrote MITRE techniques+code CSV with {len(combined_df)} rows to {output_csv}"
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        import config
    except Exception:
        config = None

    cti_dir = (
        getattr(config, "CODE_EXTRACTOR_OUTPUT_DIR", None) if config else None
    ) or os.path.join(base_dir, "data", "cti_code_only")

    code_tokens_csv = (
        getattr(config, "MITRE_CODE_TOKENS_CSV", None) if config else None
    ) or os.path.join(base_dir, "data", "reference_resources", "MitreCodeTokens_V1.csv")

    # 1) Build per-code-snippet tokens CSV (technique_name/procedure-level)
    build_mitre_code_tokens(cti_dir=cti_dir, output_csv=code_tokens_csv)

    # 2) Build a technique-level MITRE CSV where each MITRE technique's text
    #    is augmented with all associated code snippets grouped by
    #    technique_name. This lives in the main data/reference_resources
    #    directory so it can be used as an alternate MITRE source.
    root_data_dir = (
        getattr(config, "REFERENCE_RESOURCES_DIR", None) if config else None
    ) or os.path.normpath(os.path.join(base_dir, "..", "..", "data", "reference_resources"))

    mitre_original_csv = (
        getattr(config, "MITRE_TECHNIQUES_CSV", None) if config else None
    ) or os.path.join(root_data_dir, "MitreTechniquesTokens_WithDB_V1.csv")
    mitre_with_code_csv = os.path.join(root_data_dir, "MitreTechniquesTokens_WithCode_V1.csv")

    try:
        build_mitre_techniques_with_code(
            mitre_csv=mitre_original_csv,
            code_tokens_csv=code_tokens_csv,
            output_csv=mitre_with_code_csv,
        )
    except Exception as exc:
        print(f"Failed to build combined MITRE+code CSV: {exc}")
