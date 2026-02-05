"""
BGE-M3 Text Encoder Wrapper

Provides a simple interface to generate dense text embeddings using
BAAI's multilingual, multi-function BGE-M3 model.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

import logging

logger = logging.getLogger(__name__)


class BGEM3TextEncoder:
    """Produces normalized text embeddings leveraging BGE-M3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        trust_remote_code: bool = True,
        embedding_dim: int = 1024,
        document_instruction: str = "Represent this document for retrieval:",
        query_instruction: str = "Represent this query for retrieving relevant documents:",
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype if self._device == "cuda" else torch.float32
        self.embedding_dim = embedding_dim
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()
        self._doc_instruction = document_instruction.strip() if document_instruction else ""
        self._query_instruction = query_instruction.strip() if query_instruction else ""
        logger.info(
            "BGE-M3 encoder initialized: model=%s device=%s dtype=%s",
            model_name,
            self._device,
            self._dtype,
        )

    def encode(self, text: str, *, instruction: Optional[str] = None) -> torch.Tensor:
        """Return a normalized embedding for the provided text (document-style by default)."""
        logger.info("Encoding single text: length=%d", len(text or ""))
        resolved_instruction = instruction
        if resolved_instruction is None and self._doc_instruction:
            resolved_instruction = self._doc_instruction
        embedding = self.encode_batch([text], instruction=resolved_instruction)[0]
        logger.info("Single text encoding complete: dim=%d", embedding.shape[-1])
        return embedding

    def encode_batch(self, texts: Iterable[str], *, instruction: Optional[str] = None) -> List[torch.Tensor]:
        """Batch encode multiple texts, returning CPU tensor embeddings."""
        texts = list(texts)
        logger.info("Batch encoding texts: count=%d instruction=%s", len(texts), instruction)
        prefix = (instruction or "").strip()
        queries = [f"{prefix} {text}" if prefix else text for text in texts]
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

        embeddings = [emb.detach().cpu() for emb in sentence_embeddings]
        logger.info("Batch encoding complete: produced=%d dim=%d", len(embeddings), embeddings[0].shape[-1] if embeddings else 0)
        return embeddings

    def encode_document(self, text: str) -> torch.Tensor:
        """Encode a document/passage using the configured document instruction."""
        logger.info("Encoding document text")
        instruction = self._doc_instruction or None
        return self.encode(text, instruction=instruction)

    def encode_query(self, text: str) -> torch.Tensor:
        """Encode a query using the configured query instruction."""
        logger.info("Encoding query text")
        instruction = self._query_instruction or None
        return self.encode(text, instruction=instruction)


__all__ = ["BGEM3TextEncoder"]
