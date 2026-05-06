import os
import re
import sys
from collections import Counter
from typing import List, Tuple, Optional

import pandas as pd


ANTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../ANTS_Share_Preprocessing_Embedding")
)
if ANTS_DIR not in sys.path:
    sys.path.insert(0, ANTS_DIR)

from standardizer import standardize


class DrainParser:
    """
    Adapter that exposes the existing parser interface while using ANTS
    normalization underneath.
    """

    _NETWORK_OP_HINTS = (
        "tcp", "udp", "connect", "disconnect", "send", "receive", "listen", "bind"
    )
    _FILE_OP_HINTS = (
        "file", "read", "write", "create", "close", "queryinformationfile", "setinformationfile"
    )

    def __init__(self, depth: int = 4, st: float = 0.5, registry_mode: bool = False, **kwargs):
        self.depth = depth
        self.st = st
        self.registry_mode = registry_mode
        self._template_counter: Counter[str] = Counter()

    @staticmethod
    def is_registry_operation(operation: str) -> bool:
        if not operation:
            return False
        op = str(operation).strip()
        return op.startswith("Reg")

    def _infer_std_type(self, row: pd.Series, log_message: str) -> str:
        operation = str(row.get("Operation", "")).strip().lower()
        path = str(row.get("Path", "")).strip().lower()
        content = str(row.get("Content", "")).strip().lower()
        joined = f"{operation} {path} {content} {log_message.lower()}"

        if self.is_registry_operation(operation) or "\\registry\\" in path or path.startswith("hklm") or path.startswith("hkcu"):
            return "registry"

        if any(hint in joined for hint in self._NETWORK_OP_HINTS):
            return "network"

        if re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d+\b", joined):
            return "network"

        if any(hint in joined for hint in self._FILE_OP_HINTS):
            return "file"

        if re.search(r"[a-z]:\\", joined) or "/" in path or "." in os.path.basename(path):
            return "file"

        return "process"

    def _build_log_message(self, row: pd.Series, columns: List[str]) -> str:
        parts = []
        for col in columns:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip() and str(val).lower() != "nan":
                    parts.append(str(val).strip())
        return " ".join(parts)

    def _pick_target_text(self, row: pd.Series, std_type: str, fallback: str) -> str:
        if std_type in ("registry", "network", "file"):
            for col in ("Path", "Content", "Detail", "Command Line"):
                val = str(row.get(col, "")).strip()
                if val:
                    return val
        else:
            for col in ("Command Line", "Process Name", "Content", "Path"):
                val = str(row.get(col, "")).strip()
                if val:
                    return val
        return fallback

    def parse_log_row(self, row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]:
        log_message = self._build_log_message(row, columns)
        std_type = self._infer_std_type(row, log_message)
        target_text = self._pick_target_text(row, std_type, log_message)

        try:
            if std_type in ("registry", "network", "file"):
                normalized = standardize(target_text, std_type, mapping_collector=[])
            else:
                normalized = standardize(target_text, std_type)
            normalized = str(normalized) if normalized is not None else target_text
        except Exception:
            normalized = target_text

        operation = str(row.get("Operation", "")).strip()
        result = str(row.get("Result", "")).strip()
        pieces = [p for p in (operation, normalized, result) if p]
        template = " ".join(pieces) if pieces else normalized
        self._template_counter[template] += 1
        return template, [], log_message

    def get_clusters(self):
        return list(self._template_counter.items())


class RegistryDrainParser(DrainParser):
    def __init__(self, *args, **kwargs):
        kwargs["registry_mode"] = True
        super().__init__(*args, **kwargs)
