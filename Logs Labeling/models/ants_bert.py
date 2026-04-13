import os
import sys
from typing import List, Optional, Union

import numpy as np

# Add ANTS standardizer package path
ANTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../ANTS_Share_Preprocessing_Embedding")
)
if ANTS_DIR not in sys.path:
    sys.path.insert(0, ANTS_DIR)

from standardizer import standardize
from models.bert import TransformerBERTModel


class ANTSBERTModel(TransformerBERTModel):
    """In-house ANTS model wrapper with optional strict standardization."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_standardizer: bool = True,
        **kwargs,
    ):
        model_path = os.path.join(ANTS_DIR, "SecurityBERT_ANTS")
        super().__init__(model_name=model_path, cache_dir=cache_dir)
        self.use_standardizer = bool(use_standardizer)
        self._warned_standardizer_fallback = False

    def _determine_type(self, dataset_name: str) -> str:
        if not dataset_name:
            raise ValueError("dataset_name is required for ANTS standardization")

        name = dataset_name.lower()
        if "file" in name:
            return "file"
        if "registry" in name:
            return "registry"
        if "network" in name:
            return "network"
        if "process" in name:
            return "process"

        raise ValueError(f"Cannot infer ANTS standardization type from dataset: {dataset_name}")

    def _standardize_one(self, text: str, std_type: str) -> str:
        if std_type in ("registry", "network", "file"):
            normalized = standardize(text, std_type, mapping_collector=[])
        else:
            normalized = standardize(text, std_type)

        if normalized is None:
            raise ValueError(f"ANTS standardizer returned None for type={std_type}, text={text}")

        return str(normalized)

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if self.use_standardizer:
            dataset_name = kwargs.get("dataset_name", "")
            try:
                std_type = self._determine_type(dataset_name)
                final_texts = [self._standardize_one(str(t), std_type) for t in texts]
            except ValueError:
                if not self._warned_standardizer_fallback:
                    print(
                        "[Warning] ANTS standardizer fallback to raw text "
                        f"(dataset_name={dataset_name!r})"
                    )
                    self._warned_standardizer_fallback = True
                final_texts = [str(t) for t in texts]
        else:
            final_texts = [str(t) for t in texts]

        return super().embed(
            texts=final_texts,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=normalize,
        )
