"""
PaddleOCR-VL Wrapper

PaddlePaddle/PaddleOCR-VL 기반 OCR 파이프라인을 캡슐화하여
이미지에서 텍스트를 추출하고 후속 파이프라인(텍스트 임베딩 등)으로 전달한다.

PaddleOCR-VL 모델을 사용하려면 사전에 해당 weight를 다운로드 받아
`det_model_dir`, `rec_model_dir`, `cls_model_dir` 인자로 전달해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import logging

from PIL import Image, ImageDraw

try:
    from paddleocr import PaddleOCRVL
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCRVL = None

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None

logger = logging.getLogger(__name__)


@dataclass
class OCRToken:
    text: str
    score: float
    box: Sequence[Sequence[float]]


@dataclass
class OCRExecutionResult:
    tokens: List[OCRToken]
    combined_text: str
    markdown_text: Optional[str] = None
    markdown_images: Optional[List[Tuple[str, Image.Image]]] = None


class PaddleOCRVLPipeline:
    """PaddleOCR-VL 기반 OCR 추론 파이프라인."""

    def __init__(
        self,
        *,
        score_threshold: float = 0.5,
        **pipeline_kwargs,
    ) -> None:
        self._score_threshold = score_threshold
        self._use_vl = PaddleOCRVL is not None

        if self._use_vl:
            self._pipeline = PaddleOCRVL(**pipeline_kwargs)
        elif PaddleOCR is not None:  # pragma: no cover - runtime fallback
            if pipeline_kwargs:
                logger.warning(
                    "PaddleOCRVL not available, falling back to PaddleOCR with kwargs: %s",
                    pipeline_kwargs,
                )
            self._pipeline = PaddleOCR(**pipeline_kwargs)
        else:  # pragma: no cover - runtime fallback
            self._pipeline = None
            logger.warning(
                "Neither PaddleOCRVL nor PaddleOCR is installed. "
                "OCR will return empty results."
            )

    def extract(self, image_path: str) -> OCRExecutionResult:
        """이미지 경로를 입력받아 OCR 결과를 반환한다."""
        if self._pipeline is None:
            return OCRExecutionResult(tokens=[], combined_text="")
        if self._use_vl:
            ocr_outputs = self._pipeline.predict(image_path)
            return self._extract_from_vl(ocr_outputs)
        return self._extract_from_std(self._pipeline.ocr(image_path, cls=True))

    def _extract_from_vl(self, ocr_outputs) -> OCRExecutionResult:
        filtered_tokens: List[OCRToken] = []
        combined_text_parts: List[str] = []
        markdown_pages = []
        markdown_images: List[Tuple[str, Image.Image]] = []

        for doc in ocr_outputs:
            doc_dict = self._to_dict(doc)
            markdown = doc_dict.get("markdown")
            if markdown:
                markdown_pages.append(markdown)
                for path, image_obj in markdown.get("markdown_images", {}).items():
                    markdown_images.append((path, image_obj))

            for page in doc_dict.get("pages", []):
                for line in page.get("lines", []):
                    text = line.get("text", "")
                    score = float(line.get("score", 1.0))
                    if not text or score < self._score_threshold:
                        continue
                    box = line.get("bbox") or line.get("box") or line.get("polygon") or []
                    box = [list(map(float, pt)) for pt in box]
                    filtered_tokens.append(OCRToken(text=text, score=score, box=box))
                    combined_text_parts.append(text)

        markdown_text = None
        if markdown_pages:
            markdown_text = self._concatenate_markdown_pages(markdown_pages)

        combined_text = " ".join(combined_text_parts).strip()
        return OCRExecutionResult(
            tokens=filtered_tokens,
            combined_text=combined_text,
            markdown_text=markdown_text,
            markdown_images=markdown_images or None,
        )

    def _concatenate_markdown_pages(self, markdown_pages: List[dict]) -> Optional[str]:
        """Safely concatenate markdown pages when PaddleOCR-VL exposes the helper."""
        if hasattr(self._pipeline, "concatenate_markdown_pages"):
            try:
                return self._pipeline.concatenate_markdown_pages(markdown_pages)
            except Exception:  # pragma: no cover - optional path
                return None

        parts = []
        for page in markdown_pages:
            text = page.get("markdown_text")
            if text:
                parts.append(text)
        if parts:
            return "\n\n".join(parts)
        return None

    def _extract_from_std(self, ocr_outputs) -> OCRExecutionResult:
        filtered_tokens: List[OCRToken] = []
        combined_text_parts: List[str] = []

        if not ocr_outputs:
            return OCRExecutionResult(tokens=[], combined_text="")

        # paddleocr returns [results] when given a single image path
        if isinstance(ocr_outputs[0], list):
            lines = ocr_outputs[0]
        else:
            lines = ocr_outputs

        for line in lines:
            if not line or len(line) < 2:
                continue
            box, text_info = line[0], line[1]

            if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                continue
            text, score = text_info[0], float(text_info[1] or 0.0)

            if not text or score < self._score_threshold:
                continue

            box = [list(map(float, pt)) for pt in box] if box else []
            filtered_tokens.append(OCRToken(text=text, score=score, box=box))
            combined_text_parts.append(text)

        combined_text = " ".join(combined_text_parts).strip()
        return OCRExecutionResult(tokens=filtered_tokens, combined_text=combined_text)

    @staticmethod
    def _to_dict(doc) -> dict:
        if hasattr(doc, "to_dict"):
            return doc.to_dict()
        if hasattr(doc, "dict"):
            return doc.dict()
        if hasattr(doc, "__dict__"):
            return getattr(doc, "__dict__")
        return {}

    def visualize(self, image_path: str, output_path: Optional[str] = None) -> None:
        """시각화된 OCR 결과를 저장하거나 화면에 띄운다."""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        results = self.extract(image_path)

        for token in results.tokens:
            if token.box:
                polygon = [tuple(pt) for pt in token.box]
                draw.polygon(polygon, outline="red", width=2)
                draw.text((polygon[0][0], polygon[0][1] - 10), token.text, fill="red")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
        else:
            image.show()


__all__ = ["OCRToken", "OCRExecutionResult", "PaddleOCRVLPipeline"]
