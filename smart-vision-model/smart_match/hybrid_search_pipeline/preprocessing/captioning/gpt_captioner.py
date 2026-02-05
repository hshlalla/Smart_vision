"""
GPT Vision Captioning Wrapper

Provides a lightweight interface for generating descriptive captions for
industrial equipment images using OpenAI vision-capable GPT models.

This module is optional: it requires the `openai` Python package and an
`OPENAI_API_KEY` environment variable.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class GPTVLCaptioner:
    """Generate captions for images via OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        device: Optional[str] = None,
        prompt: str = (
            "Describe this industrial component in detail, including its visible shape, "
            "color, material, labels, and any text on it."
        ),
        max_image_side: int = 1024,
        image_detail: str = "low",
        timeout_s: float = 120.0,
    ) -> None:
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._prompt = (prompt or "").strip()
        self._max_image_side = int(max_image_side)
        self._image_detail = (os.getenv("OPENAI_IMAGE_DETAIL", image_detail) or "low").strip().lower()
        self._timeout_s = float(timeout_s)

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError("GPTVLCaptioner requires `openai`. Install it with `pip install openai`.") from exc

        self._client = OpenAI()
        logger.info("GPT captioner initialized: model=%s device=%s detail=%s", self._model, self._device, self._image_detail)

    def generate(self, image_path: str | Path, *, prompt: Optional[str] = None) -> str:
        start_time = time.perf_counter()
        image_path = Path(image_path)
        text_prompt = (prompt or self._prompt or "").strip()
        if not text_prompt:
            text_prompt = "Describe this image."

        image_b64 = self._encode_image(image_path)
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": self._image_detail,
                    },
                ],
            }
        ]

        try:
            response = self._client.responses.create(
                model=self._model,
                input=input_payload,
                timeout=self._timeout_s,
            )
        except TypeError:
            # Older openai clients may not accept `timeout` in create(); rely on defaults.
            response = self._client.responses.create(
                model=self._model,
                input=input_payload,
            )

        caption = getattr(response, "output_text", None)
        if caption is None:
            caption = ""

        caption = str(caption).strip()
        duration = time.perf_counter() - start_time
        logger.info("GPT caption completed: model=%s image=%s chars=%d duration=%.2fs", self._model, image_path, len(caption), duration)
        return caption

    def _encode_image(self, image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        image = self._resize(image)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _resize(self, image: Image.Image) -> Image.Image:
        if self._max_image_side <= 0:
            return image
        width, height = image.size
        max_side = max(width, height)
        if max_side <= self._max_image_side:
            return image
        scale = self._max_image_side / float(max_side)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size, Image.BICUBIC)


__all__ = ["GPTVLCaptioner"]
