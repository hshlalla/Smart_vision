"""
Qwen-VL Captioning Wrapper

Provides a lightweight interface for generating descriptive captions for
industrial equipment images using the Qwen3-VL-8B-Instruct model family.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

try:  # Optional: only available on newer `transformers` releases.
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except ImportError:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None

logger = logging.getLogger(__name__)


class Qwen3VLCaptioner:
    """Generate detailed captions for equipment imagery via Qwen3-VL."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        device: Optional[str] = None,
        dtype: torch.dtype | None = torch.float16,
        prompt: str = (
            "Describe this industrial component in detail, including its visible shape, "
            "color, material, labels, and any text on it."
        ),
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ) -> None:
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._prompt = prompt.strip()
        self._max_new_tokens = max_new_tokens
        self._do_sample = do_sample

        self._uses_chat_template = True
        primary_model = model_name
        # Keep the fallback reasonably small so demo runs don't trigger multi-GB downloads.
        fallback_model = "Qwen/Qwen2-VL-2B-Instruct"

        dtype = self._dtype if self._device == "cuda" and self._dtype is not None else torch.float32

        try:
            self._processor = AutoProcessor.from_pretrained(primary_model, trust_remote_code=True)
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if self._device == "cuda":
                load_kwargs["device_map"] = "auto"
            if Qwen3VLForConditionalGeneration is not None:
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(primary_model, **load_kwargs)
            else:
                self._model = AutoModelForVision2Seq.from_pretrained(primary_model, **load_kwargs)
            if self._device != "cuda":
                self._model.to(self._device)
            self._model_name = primary_model
            logger.info("Loaded caption model %s", primary_model)
        except (ValueError, KeyError, OSError) as exc:
            logger.warning("Failed to load %s (%s); falling back to %s", primary_model, exc, fallback_model)
            self._processor = AutoProcessor.from_pretrained(fallback_model, trust_remote_code=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                fallback_model,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self._device)
            self._uses_chat_template = False
            self._model_name = fallback_model

        self._model.eval()

    def generate(self, image_path: str | Path, *, prompt: Optional[str] = None) -> str:
        """Return a caption describing the provided image."""
        start_time = time.perf_counter()
        logger.info("Captioner invoked: model=%s image=%s", self._model_name, image_path)
        image = Image.open(image_path).convert("RGB")
        text_prompt = (prompt or self._prompt).strip()

        if self._uses_chat_template and hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
        else:
            inputs = self._processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            ).to(self._model.device)

        generation_kwargs = {
            "max_new_tokens": self._max_new_tokens,
            "do_sample": self._do_sample,
        }
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
            generation_kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )
        if "input_ids" in inputs:
            input_length = inputs["input_ids"].shape[-1]
            generated_ids = outputs[:, input_length:] if outputs.shape[-1] > input_length else outputs
        else:
            generated_ids = outputs

        captions = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        caption = captions[0] if captions else ""
        total_duration = time.perf_counter() - start_time
        logger.info(
            "Captioner completed: model=%s image=%s chars=%d duration=%.2fs",
            self._model_name,
            image_path,
            len(caption),
            total_duration,
        )
        if caption:
            logger.info(
                "Caption output for %s: %s",
                image_path,
                caption.replace("\n", " "),
            )
        else:
            logger.info("Caption output for %s is empty.", image_path)
        return caption.strip()


__all__ = ["Qwen3VLCaptioner"]
