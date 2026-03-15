"""
Qwen3-VL Embedding Wrappers

Provides shared image/text encoders backed by the Qwen3-VL-Embedding-2B model.
The wrappers expose the same encode/encode_query/encode_document interface used
by the existing pipeline so the orchestrator can swap from BGE models without
changing higher-level logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import torch

try:  # pragma: no cover - runtime dependency
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel  # type: ignore
except Exception as exc:  # pragma: no cover
    Qwen3VLModel = None
    _QWEN3_VL_MODEL_IMPORT_ERROR = exc
else:  # pragma: no cover
    _QWEN3_VL_MODEL_IMPORT_ERROR = None

try:  # pragma: no cover - runtime dependency
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # type: ignore
except Exception as exc:  # pragma: no cover
    Qwen3VLProcessor = None
    _QWEN3_VL_PROCESSOR_IMPORT_ERROR = exc
else:  # pragma: no cover
    _QWEN3_VL_PROCESSOR_IMPORT_ERROR = None

try:  # pragma: no cover - runtime dependency
    from qwen_vl_utils.vision_process import process_vision_info  # type: ignore
except Exception as exc:  # pragma: no cover
    _QWEN3_VL_UTILS_IMPORT_ERROR = exc
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except Exception as inner_exc:  # pragma: no cover
        process_vision_info = None
        _QWEN3_VL_UTILS_IMPORT_ERROR = inner_exc
    else:  # pragma: no cover
        _QWEN3_VL_UTILS_IMPORT_ERROR = None
else:  # pragma: no cover
    _QWEN3_VL_UTILS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

_BACKENDS: dict[tuple, "Qwen3VLEmbeddingBackend"] = {}


def _resolve_embedding_dim(model) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Unable to determine Qwen3-VL embedding dimension: missing model config.")
    for attr in ("hidden_size", "projection_dim"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    for config_attr in ("text_config", "vision_config"):
        child = getattr(config, config_attr, None)
        if child is None:
            continue
        value = getattr(child, "hidden_size", None)
        if value is not None:
            return int(value)
    raise ValueError("Unable to determine Qwen3-VL embedding dimension automatically.")


class Qwen3VLEmbeddingBackend:
    def __init__(
        self,
        *,
        model_name: str,
        device: Optional[str],
        dtype: Optional[torch.dtype],
        max_length: int,
    ) -> None:
        if Qwen3VLModel is None or Qwen3VLProcessor is None or process_vision_info is None:
            details = []
            if _QWEN3_VL_MODEL_IMPORT_ERROR is not None:
                details.append(f"model import failed: {type(_QWEN3_VL_MODEL_IMPORT_ERROR).__name__}: {_QWEN3_VL_MODEL_IMPORT_ERROR}")
            if _QWEN3_VL_PROCESSOR_IMPORT_ERROR is not None:
                details.append(
                    f"processor import failed: {type(_QWEN3_VL_PROCESSOR_IMPORT_ERROR).__name__}: {_QWEN3_VL_PROCESSOR_IMPORT_ERROR}"
                )
            if _QWEN3_VL_UTILS_IMPORT_ERROR is not None:
                details.append(
                    f"qwen-vl-utils import failed: {type(_QWEN3_VL_UTILS_IMPORT_ERROR).__name__}: {_QWEN3_VL_UTILS_IMPORT_ERROR}"
                )
            detail_text = f" Details: {'; '.join(details)}" if details else ""
            raise ImportError(
                "Qwen3-VL embedding requires transformers>=4.57 with Qwen3-VL support and qwen-vl-utils installed."
                + detail_text
            )

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype if self._device == "cuda" and dtype is not None else torch.float32
        self._processor = Qwen3VLProcessor.from_pretrained(model_name)
        load_kwargs = {"torch_dtype": self._dtype}
        if self._device == "cuda":
            load_kwargs["device_map"] = "auto"
        self._model = Qwen3VLModel.from_pretrained(model_name, **load_kwargs)
        if self._device != "cuda":
            self._model.to(self._device)
        self._model.eval()
        self._max_length = max_length
        self.embedding_dim = _resolve_embedding_dim(self._model)
        logger.info(
            "Qwen3-VL embedder initialized: model=%s device=%s dtype=%s embedding_dim=%d",
            model_name,
            self._device,
            self._dtype,
            self.embedding_dim,
        )

    def encode_items(self, items: List[dict]) -> List[torch.Tensor]:
        if not items:
            return []

        messages = [self._build_message(item) for item in items]
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
            embeddings = self._last_token_pool(outputs.last_hidden_state, inputs.attention_mask)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return [row.detach().cpu() for row in embeddings]

    @staticmethod
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = seq_lengths.clamp(min=0)
        batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        return last_hidden_state[batch_indices, seq_lengths]

    @staticmethod
    def _build_message(item: dict) -> dict:
        instruction = str(item.get("instruction") or "").strip()
        text = str(item.get("text") or "").strip()
        image = str(item.get("image") or "").strip()

        content = []
        if image:
            content.append({"type": "image", "image": image})
        combined_text = "\n".join(part for part in (instruction, text) if part).strip()
        if combined_text:
            content.append({"type": "text", "text": combined_text})
        if not content:
            content.append({"type": "text", "text": instruction or "Represent this input for retrieval."})
        return {"role": "user", "content": content}


def _get_backend(
    *,
    model_name: str,
    device: Optional[str],
    dtype: Optional[torch.dtype],
    max_length: int,
) -> Qwen3VLEmbeddingBackend:
    key = (model_name, device or "", str(dtype), max_length)
    backend = _BACKENDS.get(key)
    if backend is None:
        backend = Qwen3VLEmbeddingBackend(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
        _BACKENDS[key] = backend
    return backend


class Qwen3VLImageEncoder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        max_length: int = 8192,
        image_instruction: str = "Represent this image for multimodal retrieval.",
    ) -> None:
        self._backend = _get_backend(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
        self.embedding_dim = self._backend.embedding_dim
        self._instruction = image_instruction.strip()

    def encode(self, image_path: str | Path) -> torch.Tensor:
        return self._backend.encode_items(
            [{"image": str(image_path), "instruction": self._instruction}]
        )[0]


class Qwen3VLTextEncoder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        max_length: int = 8192,
        document_instruction: str = "Represent this document for multimodal retrieval.",
        query_instruction: str = "Represent this query for multimodal retrieval.",
    ) -> None:
        self._backend = _get_backend(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
        self.embedding_dim = self._backend.embedding_dim
        self._doc_instruction = document_instruction.strip()
        self._query_instruction = query_instruction.strip()

    def encode(self, text: str, *, instruction: Optional[str] = None) -> torch.Tensor:
        return self.encode_batch([text], instruction=instruction)[0]

    def encode_batch(self, texts: Iterable[str], *, instruction: Optional[str] = None) -> List[torch.Tensor]:
        resolved_instruction = (instruction or "").strip()
        items = [{"text": str(text or ""), "instruction": resolved_instruction} for text in texts]
        return self._backend.encode_items(items)

    def encode_document(self, text: str) -> torch.Tensor:
        return self.encode(text, instruction=self._doc_instruction)

    def encode_query(self, text: str) -> torch.Tensor:
        return self.encode(text, instruction=self._query_instruction)


__all__ = ["Qwen3VLImageEncoder", "Qwen3VLTextEncoder"]
