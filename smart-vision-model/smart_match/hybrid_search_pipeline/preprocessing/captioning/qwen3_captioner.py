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
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:  # Optional: only available on newer `transformers` releases.
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except ImportError:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None

try:  # Optional: only available on newer `transformers` releases.
    from transformers import Qwen2VLForConditionalGeneration  # type: ignore
except ImportError:  # pragma: no cover
    Qwen2VLForConditionalGeneration = None

try:  # Optional: for Qwen2/Qwen3 processor construction.
    from transformers import AutoVideoProcessor  # type: ignore
except ImportError:  # pragma: no cover
    AutoVideoProcessor = None

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
            self._processor = self._load_processor(primary_model)
            load_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if self._device == "cuda":
                load_kwargs["device_map"] = "auto"
            self._model = self._load_model(primary_model, load_kwargs)
            if self._device != "cuda":
                self._model.to(self._device)
            self._model_name = primary_model
            logger.info("Loaded caption model %s", primary_model)
        except Exception as exc:  # pragma: no cover - runtime dependency variability
            logger.warning("Failed to load %s (%s); falling back to %s", primary_model, exc, fallback_model)
            try:
                self._processor = self._load_processor(fallback_model)
                self._model = self._load_model(
                    fallback_model,
                    {"torch_dtype": dtype, "trust_remote_code": True},
                ).to(self._device)
                self._model_name = fallback_model
            except Exception as fallback_exc:
                logger.warning("Failed to load fallback caption model %s (%s); disabling captioner", fallback_model, fallback_exc)
                self._processor = None
                self._model = None
                self._model_name = "disabled"

        if self._model is not None:
            self._model.eval()

    @staticmethod
    def _load_model(model_name: str, load_kwargs: dict) -> torch.nn.Module:
        """Load a Qwen-VL model with compatibility across transformers versions."""
        if "Qwen3-VL" in model_name and Qwen3VLForConditionalGeneration is not None:
            return Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        if "Qwen2-VL" in model_name and Qwen2VLForConditionalGeneration is not None:
            return Qwen2VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

        if Qwen3VLForConditionalGeneration is not None:
            return Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        if Qwen2VLForConditionalGeneration is not None:
            return Qwen2VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

        # Last resort: rely on remote code registering a compatible causal LM.
        return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    @staticmethod
    def _load_processor(model_name: str):
        """Load a processor while avoiding known AutoProcessor issues for Qwen-VL in some envs."""
        if "Qwen3-VL" in model_name:
            from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

            return Qwen3VLCaptioner._build_qwen_processor(Qwen3VLProcessor, model_name)
        if "Qwen2-VL" in model_name:
            from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

            return Qwen3VLCaptioner._build_qwen_processor(Qwen2VLProcessor, model_name)

        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    @staticmethod
    def _build_qwen_processor(processor_cls, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        chat_template = getattr(tokenizer, "chat_template", None)

        if AutoVideoProcessor is None:
            raise ImportError("AutoVideoProcessor unavailable; install torchvision to enable Qwen-VL captioning.")
        video_processor = AutoVideoProcessor.from_pretrained(model_name, trust_remote_code=True)

        return processor_cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )

    def generate(self, image_path: str | Path, *, prompt: Optional[str] = None) -> str:
        """Return a caption describing the provided image."""
        if self._processor is None or self._model is None:
            return ""
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
