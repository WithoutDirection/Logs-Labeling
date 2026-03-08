"""
Reference Builder
=================
Scans a directory of CSV source files, normalises them to a common schema,
cleans/tokenises their text via TextProcessor, deduplicates by technique_id
(concatenating descriptions from multiple sources for the same technique),
and writes a single ``combined.csv`` that is consumed by the embedding and
TF-IDF build steps.

Usage (standalone):
    python reference_builder.py                     # uses config defaults
    python reference_builder.py --sources-dir /my/csvs --output /my/combined.csv

Usage (programmatic):
    from external_sources.reference_builder import ReferenceBuilder
    rb = ReferenceBuilder()
    combined_csv = rb.build()   # returns output path
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
CURRENT_DIR = str(Path(__file__).resolve().parent)
PROJECT_ROOT = str(Path(CURRENT_DIR).parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

# ---------------------------------------------------------------------------
# Constants: canonical column names every source must map to
# ---------------------------------------------------------------------------
COL_TECH_ID   = "technique_id"   # required
COL_TECH_NAME = "technique"      # required
COL_DESC_RAW  = "description_raw"
COL_DESC      = "description"
COL_TOKENS    = "tokens"
COL_CLEANED   = "cleaned_tokens"
COL_EXT_ID    = "external_id"    # optional Txxxx mapping

# Alternative column names we'll accept from source CSVs
_ID_ALIASES   = ["technique_id", "id", "stix_id", "tactic_id"]
_NAME_ALIASES = ["technique", "name", "technique_name", "tactic_name", "title"]
_DESC_ALIASES = ["description_raw", "all_text", "description_clean",
                 "description", "content", "text", "summary", "detail"]


class ReferenceBuilder:
    """
    Manages a directory of reference CSV files and produces a single
    cleaned, tokenised, deduplicated ``combined.csv``.

    Deduplication strategy:
        Multiple sources that share the same ``technique_id`` have their
        description texts **concatenated** (separated by double newline)
        before tokenisation, so that richer context from each source is
        preserved.

    Args:
        sources_dir: Directory containing one or more source CSV files.
                     Defaults to ``config.REFERENCE_SOURCES_DIR``.
        output_csv:  Path for the combined output CSV.
                     Defaults to ``config.REFERENCE_COMBINED_CSV``.
        bert_model:  BERT model name passed to TextProcessor.
                     Defaults to ``config.BERT_MODEL_NAME``.
        force_rebuild: If False, skip rebuild when the output already
                       exists and is newer than all source files.
    """

    def __init__(
        self,
        sources_dir: Optional[str] = None,
        output_csv: Optional[str] = None,
        bert_model: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        self.sources_dir  = sources_dir  or getattr(config, "REFERENCE_SOURCES_DIR",
                                                      os.path.join(config.REFERENCE_RESOURCES_DIR, "sources"))
        self.output_csv   = output_csv   or getattr(config, "REFERENCE_COMBINED_CSV",
                                                      os.path.join(config.REFERENCE_RESOURCES_DIR, "combined.csv"))
        self.bert_model   = bert_model   or config.BERT_MODEL_NAME
        self.force_rebuild = force_rebuild

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> str:
        """
        Full pipeline: scan → load → normalise → merge → clean → write.

        Returns the path to the written ``combined.csv``.
        """
        csv_files = self._scan()
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in sources directory: {self.sources_dir}"
            )

        if not self.force_rebuild and self._is_up_to_date(csv_files):
            print(f"[ReferenceBuilder] combined.csv is up-to-date, skipping rebuild.")
            return self.output_csv

        print(f"[ReferenceBuilder] Found {len(csv_files)} source file(s) in {self.sources_dir}")

        raw_frames: list[pd.DataFrame] = []
        for path in csv_files:
            df = self._load_and_normalise(path)
            if df is not None and not df.empty:
                raw_frames.append(df)
                print(f"  + {os.path.basename(path)}: {len(df)} rows")

        if not raw_frames:
            raise ValueError("All source CSVs were empty or could not be normalised.")

        merged = self._merge(raw_frames)
        print(f"[ReferenceBuilder] After merge/dedup: {len(merged)} unique techniques")

        combined = self._tokenise(merged)

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        combined.to_csv(self.output_csv, index=False)
        print(f"[ReferenceBuilder] Written combined.csv → {self.output_csv}")
        return self.output_csv

    # ------------------------------------------------------------------
    # Step 1: scan
    # ------------------------------------------------------------------

    def _scan(self) -> List[str]:
        """Return sorted list of .csv paths inside sources_dir."""
        if not os.path.isdir(self.sources_dir):
            return []
        return sorted(
            os.path.join(self.sources_dir, f)
            for f in os.listdir(self.sources_dir)
            if f.lower().endswith(".csv")
        )

    # ------------------------------------------------------------------
    # Step 2: load & normalise one CSV
    # ------------------------------------------------------------------

    def _load_and_normalise(self, path: str) -> Optional[pd.DataFrame]:
        """
        Load a CSV and map its columns to the canonical schema.

        Returns a DataFrame with at least ``technique_id``, ``technique``,
        and ``description_raw`` columns, or None on failure.
        """
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  [Warning] Could not read {path}: {exc}")
            return None

        if df.empty:
            return None

        cols_lower = {c.lower().strip(): c for c in df.columns}

        # --- Map technique_id ---
        id_col = self._find_col(cols_lower, _ID_ALIASES)
        if id_col is None:
            print(f"  [Warning] No technique_id column found in {os.path.basename(path)}, skipping.")
            return None
        df = df.rename(columns={id_col: COL_TECH_ID})
        df[COL_TECH_ID] = df[COL_TECH_ID].fillna("").astype(str).str.strip()

        # --- Map technique name ---
        name_col = self._find_col(cols_lower, _NAME_ALIASES, exclude={id_col})
        if name_col:
            df = df.rename(columns={name_col: COL_TECH_NAME})
        else:
            df[COL_TECH_NAME] = df[COL_TECH_ID]  # fallback: use ID as name

        # --- Map description ---
        desc_col = self._find_col(cols_lower, _DESC_ALIASES,
                                   exclude={id_col, name_col} if name_col else {id_col})
        if desc_col:
            df = df.rename(columns={desc_col: COL_DESC_RAW})
        else:
            # Concatenate all remaining text columns as description
            used = {COL_TECH_ID, COL_TECH_NAME}
            text_cols = [c for c in df.columns if c not in used and df[c].dtype == object]
            if text_cols:
                df[COL_DESC_RAW] = df[text_cols].fillna("").agg(" ".join, axis=1)
            else:
                df[COL_DESC_RAW] = ""

        # --- Carry over external_id if present ---
        ext_col = self._find_col(cols_lower, ["external_id", "ext_id", "mitre_id"])
        if ext_col and ext_col != COL_EXT_ID:
            df = df.rename(columns={ext_col: COL_EXT_ID})

        keep = [COL_TECH_ID, COL_TECH_NAME, COL_DESC_RAW]
        if COL_EXT_ID in df.columns:
            keep.append(COL_EXT_ID)

        return df[keep].copy()

    # ------------------------------------------------------------------
    # Step 3: merge & deduplicate
    # ------------------------------------------------------------------

    def _merge(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate all source frames and deduplicate by ``technique_id``.
        For the same technique_id, description_raw texts are joined with
        a double newline so no information is lost.
        """
        all_rows = pd.concat(frames, ignore_index=True)

        # Drop rows with empty technique_id
        all_rows = all_rows[all_rows[COL_TECH_ID].str.len() > 0]

        def _join_unique(series: pd.Series) -> str:
            parts = []
            seen: set[str] = set()
            for val in series:
                s = str(val).strip()
                if s and s not in seen:
                    seen.add(s)
                    parts.append(s)
            return "\n\n".join(parts)

        agg: dict = {
            COL_TECH_NAME: "first",
            COL_DESC_RAW:  _join_unique,
        }
        if COL_EXT_ID in all_rows.columns:
            agg[COL_EXT_ID] = "first"

        merged = (
            all_rows.groupby(COL_TECH_ID, sort=False, as_index=False)
            .agg(agg)
            .reset_index(drop=True)
        )
        return merged

    # ------------------------------------------------------------------
    # Step 4: clean & tokenise
    # ------------------------------------------------------------------

    def _tokenise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run TextProcessor over description_raw to produce:
          - description      (cleaned text, same as description_raw here)
          - description_clean
          - tokens
          - cleaned_tokens
        """
        from external_sources.text_processor import TextProcessor

        texts = df[COL_DESC_RAW].fillna("").astype(str).tolist()

        tp = TextProcessor(bert_model_name=self.bert_model)
        tp.fit_zipf_filter(texts)

        descriptions_clean: list[str] = []
        tokens_list: list[list] = []
        cleaned_tokens_list: list[list] = []

        for text in texts:
            descriptions_clean.append(tp.clean_text(text))
            tokens_list.append(tp.tokenize(text, remove_stopwords=False, apply_zipf_filter=False))
            cleaned_tokens_list.append(tp.tokenize(text, remove_stopwords=True, apply_zipf_filter=True))

        out = df.copy()
        out[COL_DESC]    = df[COL_DESC_RAW]   # keep raw as primary description
        out["description_clean"] = descriptions_clean
        out[COL_TOKENS]  = tokens_list
        out[COL_CLEANED] = cleaned_tokens_list
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_col(cols_lower: dict, aliases: list, exclude: set = None) -> Optional[str]:
        """
        Return the *original* column name whose lowercased version matches
        the first alias found, excluding already-assigned columns.
        """
        exclude = exclude or set()
        for alias in aliases:
            original = cols_lower.get(alias.lower())
            if original and original not in exclude:
                return original
        return None

    def _is_up_to_date(self, csv_files: List[str]) -> bool:
        """True if output exists and is newer than all source files."""
        if not os.path.exists(self.output_csv):
            return False
        out_mtime = os.path.getmtime(self.output_csv)
        return all(os.path.getmtime(f) < out_mtime for f in csv_files)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build combined reference CSV from a directory of source CSVs."
    )
    parser.add_argument(
        "--sources-dir", default=None,
        help="Directory containing source CSV files (default: config.REFERENCE_SOURCES_DIR)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output combined CSV path (default: config.REFERENCE_COMBINED_CSV)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild even if output is up-to-date"
    )
    args = parser.parse_args()

    rb = ReferenceBuilder(
        sources_dir=args.sources_dir,
        output_csv=args.output,
        force_rebuild=args.force,
    )
    rb.build()


if __name__ == "__main__":
    main()
