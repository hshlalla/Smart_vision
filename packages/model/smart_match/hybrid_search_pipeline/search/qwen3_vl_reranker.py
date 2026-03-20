"""
Qwen3-VL reranker wrapper.

Uses the official Qwen3-VL reranker loading pattern:
load the conditional-generation model, reuse its backbone, and score the last
token hidden state with a yes/no linear head derived from the LM head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import torch
from PIL import Image
from smart_match.device_utils import preferred_inference_dtype, preferred_torch_device
from transformers import AutoProcessor

try:  # pragma: no cover - runtime dependency
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except Exception:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None

try:  # pragma: no cover - runtime dependency
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:  # pragma: no cover
    process_vision_info = None

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."


class Qwen3VLReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Reranker-2B",
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = torch.float16,
        max_length: int = 2048,
        instruction: str = DEFAULT_INSTRUCTION,
        batch_size: int = 1,
        max_image_side: int = 384,
    ) -> None:
        if Qwen3VLForConditionalGeneration is None or process_vision_info is None:
            raise ImportError(
                "Qwen3-VL reranker requires transformers>=4.57 with Qwen3-VL support and qwen-vl-utils installed."
            )
        self._device = preferred_torch_device(device)
        self._dtype = preferred_inference_dtype(self._device, dtype)
        self._max_length = max_length
        self._instruction = instruction.strip() or DEFAULT_INSTRUCTION
        self._batch_size = max(1, int(batch_size))
        self._max_image_side = max(128, int(max_image_side))
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
            use_fast=False,
        )
        load_kwargs: dict[str, Any] = {
            "dtype": self._dtype,
            "trust_remote_code": True,
        }
        if self._device == "cuda":
            load_kwargs["device_map"] = "auto"
        lm = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        self._generator = lm
        self._model = lm.model
        if self._device != "cuda":
            self._generator.to(self._device)
        self._generator.eval()
        self._model.eval()
        self._score_linear = self._build_binary_head(lm)
        self._score_linear.eval()
        logger.info(
            "Qwen3-VL reranker initialized: model=%s device=%s dtype=%s",
            model_name,
            self._device,
            self._dtype,
        )

    def score(self, query: dict, candidates: List[dict]) -> List[float]:
        if not candidates:
            return []
        pairs = [self._build_pair(query, candidate) for candidate in candidates]
        scores: list[float] = []
        for start in range(0, len(pairs), self._batch_size):
            batch_pairs = pairs[start : start + self._batch_size]
            batch_candidates = candidates[start : start + self._batch_size]
            try:
                inputs = self._tokenize_pairs(batch_pairs)
                with torch.no_grad():
                    hidden = self._model(**inputs).last_hidden_state[:, -1]
                    logits = self._score_linear(hidden)
                    batch_scores = torch.sigmoid(logits).squeeze(-1).detach().cpu().tolist()
                if not isinstance(batch_scores, list):
                    batch_scores = [float(batch_scores)]
                scores.extend(float(score) for score in batch_scores)
            except Exception:
                logger.exception(
                    "Qwen3-VL reranker scoring failed for candidates=%s",
                    [candidate.get("model_id") for candidate in batch_candidates],
                )
                scores.extend(0.0 for _ in batch_candidates)
        return scores

    def _build_binary_head(self, model: Qwen3VLForConditionalGeneration) -> torch.nn.Linear:
        vocab = self._processor.tokenizer.get_vocab()
        token_yes = vocab["yes"]
        token_no = vocab["no"]
        lm_head_weights = model.lm_head.weight.data
        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]
        dim = int(weight_yes.shape[0])
        linear = torch.nn.Linear(dim, 1, bias=False)
        with torch.no_grad():
            linear.weight[0] = weight_yes - weight_no
        linear = linear.to(self._device)
        linear = linear.to(self._model.dtype)
        return linear

    def _tokenize_pairs(self, pairs: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
        text = self._processor.apply_chat_template(
            pairs,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos, video_kwargs = process_vision_info(
            pairs,
            return_video_kwargs=True,
            image_patch_size=16,
        )
        inputs = self._processor(
            text=text,
            images=images,
            videos=videos,
            truncation=False,
            padding=False,
            do_resize=False,
            **(video_kwargs or {}),
        )
        special_token_ids = set(self._processor.tokenizer.all_special_ids)
        input_ids = inputs["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            for index, token_ids in enumerate(input_ids):
                tail_size = 5 if len(token_ids) > 5 else 0
                head = token_ids[:-tail_size] if tail_size else token_ids
                tail = token_ids[-tail_size:] if tail_size else []
                trimmed = self._truncate_token_ids(head, self._max_length - len(tail), special_token_ids)
                input_ids[index] = trimmed + tail

        padded = self._processor.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
            max_length=self._max_length,
        )
        for key, value in padded.items():
            inputs[key] = value
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                continue
            if hasattr(value, "shape"):
                inputs[key] = torch.as_tensor(value)
                continue
            if isinstance(value, list) and value and not isinstance(value[0], str):
                inputs[key] = torch.as_tensor(value)
        return inputs.to(self._device)

    @staticmethod
    def _truncate_token_ids(token_ids: list[int], max_length: int, special_token_ids: set[int]) -> list[int]:
        if len(token_ids) <= max_length:
            return token_ids
        special_count = sum(1 for token_id in token_ids if token_id in special_token_ids)
        non_special_budget = max(0, max_length - special_count)
        kept: list[int] = []
        non_special_kept = 0
        for token_id in token_ids:
            if token_id in special_token_ids:
                kept.append(token_id)
                continue
            if non_special_kept < non_special_budget:
                kept.append(token_id)
                non_special_kept += 1
        return kept

    def _build_pair(self, query: dict, candidate: dict) -> list[dict[str, Any]]:
        query_text = str(query.get("text") or "").strip()
        query_image = self._load_image(query.get("image"))
        candidate_text = str(candidate.get("aggregated_text") or candidate.get("description") or "").strip()
        candidate_image = None
        candidate_images = candidate.get("images") if isinstance(candidate.get("images"), list) else []
        if candidate_images:
            candidate_image = self._load_image(candidate_images[0].get("image_path"))

        contents: list[dict[str, Any]] = [
            {"type": "text", "text": f"<Instruct>: {self._instruction}"},
            {"type": "text", "text": "<Query>:"},
        ]
        if query_image:
            contents.append({"type": "image", "image": query_image})
        contents.append({"type": "text", "text": query_text or "NULL"})
        contents.append({"type": "text", "text": "\n<Document>:"})
        if candidate_image:
            contents.append({"type": "image", "image": candidate_image})
        contents.append({"type": "text", "text": candidate_text or "NULL"})

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": contents,
            },
        ]

    @staticmethod
    def _resolve_media_path(value: object) -> str:
        path = str(value or "").strip()
        if not path:
            return ""
        if path.startswith(("http://", "https://", "data:")):
            return path
        resolved = Path(path).expanduser()
        if resolved.exists():
            return str(resolved.resolve())
        repo_root = Path(__file__).resolve().parents[5]
        for anchor in ("data", "media"):
            parts = resolved.parts
            if anchor in parts:
                rewritten = repo_root.joinpath(*parts[parts.index(anchor) :])
                if rewritten.exists():
                    return str(rewritten.resolve())
        return path

    def _load_image(self, value: object) -> Image.Image | str | None:
        path = self._resolve_media_path(value)
        if not path:
            return None
        if path.startswith(("http://", "https://", "data:")):
            return path
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((self._max_image_side, self._max_image_side), Image.Resampling.LANCZOS)
                return image.copy()
        except Exception:
            logger.exception("Failed to prepare reranker image: path=%s", path)
            return None


__all__ = ["Qwen3VLReranker"]
