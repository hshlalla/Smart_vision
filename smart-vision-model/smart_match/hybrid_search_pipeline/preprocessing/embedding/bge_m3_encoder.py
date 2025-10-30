"""
BGE-M3 Text Encoder Wrapper

Provides a simple interface to generate dense text embeddings using
BAAI's multilingual, multi-function BGE-M3 model.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer


class BGEM3TextEncoder:
    """Produces normalized text embeddings leveraging BGE-M3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        trust_remote_code: bool = True,
        embedding_dim: int = 1024,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch_dtype
        self.embedding_dim = embedding_dim
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()

    def encode(self, text: str, *, instruction: Optional[str] = None) -> torch.Tensor:
        """Return a normalized embedding for the provided text."""
        return self.encode_batch([text], instruction=instruction)[0]

    def encode_batch(self, texts: Iterable[str], *, instruction: Optional[str] = None) -> List[torch.Tensor]:
        """Batch encode multiple texts, returning CPU tensor embeddings."""
        prefix = instruction or ""
        queries = [f"{prefix}{text}" if prefix else text for text in texts]
        inputs = self._tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs.attention_mask.unsqueeze(-1)
            masked_embeddings = token_embeddings * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-8)
            sentence_embeddings = sum_embeddings / sum_mask
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=-1)

        return [emb.detach().cpu() for emb in sentence_embeddings]


__all__ = ["BGEM3TextEncoder"]
