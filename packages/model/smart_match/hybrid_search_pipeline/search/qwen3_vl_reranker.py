"""
Qwen3-VL Reranker Wrapper

Reorders top hybrid-search candidates using the Qwen3-VL-Reranker-2B model.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from transformers import AutoModelForSequenceClassification

try:  # pragma: no cover - runtime dependency
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # type: ignore
except Exception:  # pragma: no cover
    Qwen3VLProcessor = None

try:  # pragma: no cover - runtime dependency
    from qwen_vl_utils.vision_process import process_vision_info  # type: ignore
except Exception:  # pragma: no cover
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except Exception:  # pragma: no cover
        process_vision_info = None

logger = logging.getLogger(__name__)


class Qwen3VLReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        max_length: int = 8192,
    ) -> None:
        if Qwen3VLProcessor is None or process_vision_info is None:
            raise ImportError(
                "Qwen3-VL reranker requires transformers>=4.57 with Qwen3-VL support and qwen-vl-utils installed."
            )
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype if self._device == "cuda" and dtype is not None else torch.float32
        self._processor = Qwen3VLProcessor.from_pretrained(model_name)
        load_kwargs = {"torch_dtype": self._dtype, "trust_remote_code": True}
        if self._device == "cuda":
            load_kwargs["device_map"] = "auto"
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)
        if self._device != "cuda":
            self._model.to(self._device)
        self._model.eval()
        self._max_length = max_length
        logger.info("Qwen3-VL reranker initialized: model=%s device=%s", model_name, self._device)

    def score(self, query: dict, candidates: List[dict]) -> List[float]:
        if not candidates:
            return []
        messages = [self._build_message(query, candidate) for candidate in candidates]
        texts = [
            self._processor.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=False,
            )
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                return [0.0 for _ in candidates]
            logits = logits.view(-1).detach().cpu().tolist()
        return [float(score) for score in logits]

    @staticmethod
    def _build_message(query: dict, candidate: dict) -> dict:
        query_text = str(query.get("text") or "").strip()
        query_image = str(query.get("image") or "").strip()
        candidate_text = str(candidate.get("aggregated_text") or candidate.get("description") or "").strip()
        candidate_images = candidate.get("images") if isinstance(candidate.get("images"), list) else []
        candidate_image = ""
        if candidate_images:
            candidate_image = str(candidate_images[0].get("image_path") or "").strip()

        content = []
        if query_image:
            content.append({"type": "image", "image": query_image})
        if candidate_image:
            content.append({"type": "image", "image": candidate_image})

        prompt_parts = [
            "Rate how relevant the candidate is to the query for retrieval. Higher is better.",
        ]
        if query_text:
            prompt_parts.append(f"Query: {query_text}")
        if candidate_text:
            prompt_parts.append(f"Candidate: {candidate_text}")
        content.append({"type": "text", "text": "\n".join(prompt_parts)})
        return {"role": "user", "content": content}


__all__ = ["Qwen3VLReranker"]
