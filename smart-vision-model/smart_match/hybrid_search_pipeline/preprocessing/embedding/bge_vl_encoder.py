"""
BGE-VL Vision Encoder Wrapper

Wraps the BAAI BGE-VL vision-language model so the preprocessing pipeline can
extract dense image embeddings for hybrid search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import logging

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


class BGEVLImageEncoder:
    """Utility for producing dense image vectors from BGE-VL."""

    def __init__(
        self,
        model_name: str = "BAAI/BGE-VL-large",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        trust_remote_code: bool = True,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype if self._device == "cuda" else torch.float32

        # Some fast image processors require torchvision. Fall back to the slow processor when
        # torchvision isn't installed (common on minimal CPU environments).
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        except ImportError as exc:
            logger.warning("Fast processor unavailable (%s); falling back to use_fast=False", exc)
            self._processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                use_fast=False,
            )
        self._model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            trust_remote_code=trust_remote_code,
        ).to(self._device)
        self._model.eval()
        self.embedding_dim = self._infer_embedding_dim(embedding_dim)
        logger.info(
            "BGE-VL encoder initialized: model=%s device=%s dtype=%s embed_dim=%s",
            model_name,
            self._device,
            self._dtype,
            self.embedding_dim,
        )

    def encode(self, image_path: str | Path) -> torch.Tensor:
        """Return a normalized embedding vector for the provided image."""
        logger.info("Encoding image: path=%s", image_path)
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)

        vector = image_features.squeeze(0).detach().cpu()
        if vector.shape[-1] != self.embedding_dim:
            self.embedding_dim = vector.shape[-1]
        logger.info("Image encoding complete: dim=%d", vector.shape[-1])

        return vector

    def _infer_embedding_dim(self, override_dim: Optional[int]) -> int:
        if override_dim is not None:
            return override_dim

        projection = getattr(self._model, "visual_projection", None)
        if projection is not None and hasattr(projection, "out_features"):
            return int(projection.out_features)

        config = getattr(self._model, "config", None)
        if config is not None:
            for attr in ("projection_dim", "hidden_size", "vision_embed_dim"):
                value = getattr(config, attr, None)
                if value is not None:
                    return int(value)

        raise ValueError("Unable to determine BGE-VL embedding dimension automatically.")


__all__ = ["BGEVLImageEncoder"]
