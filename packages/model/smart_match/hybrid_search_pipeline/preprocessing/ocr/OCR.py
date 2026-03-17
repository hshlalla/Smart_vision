"""
PaddleOCR-VL Wrapper

PaddlePaddle/PaddleOCR-VL 기반 OCR 파이프라인을 캡슐화하여
이미지에서 텍스트를 추출하고 후속 파이프라인(텍스트 임베딩 등)으로 전달한다.

PaddleOCR-VL 모델을 사용하려면 사전에 해당 weight를 다운로드 받아
`det_model_dir`, `rec_model_dir`, `cls_model_dir` 인자로 전달해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import logging
from importlib import import_module

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


def _supports_paddleocr_vl() -> bool:
    """Check whether the installed Paddle runtime exposes PaddleOCR-VL fused ops."""
    try:
        functional = import_module("paddle.incubate.nn.functional")
    except Exception:
        return False
    return hasattr(functional, "fused_rms_norm_ext")


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
    structured_text: Optional[str] = None
    spotting_text: str = ""
    table_htmls: Dict[str, str] = field(default_factory=dict)
    block_labels: List[str] = field(default_factory=list)
    block_contents: List[str] = field(default_factory=list)


class PaddleOCRVLPipeline:
    """PaddleOCR-VL 기반 OCR 추론 파이프라인."""

    def __init__(
        self,
        *,
        score_threshold: float = 0.5,
        **pipeline_kwargs,
    ) -> None:
        self._score_threshold = score_threshold
        self._pipeline_kwargs = dict(pipeline_kwargs)
        self._use_vl = False
        self._pipeline = None

        pipeline_defaults = {
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": True,
            "use_layout_detection": True,
            "use_chart_recognition": True,
            "use_seal_recognition": True,
            "use_ocr_for_image_block": True,
            "format_block_content": True,
            "merge_layout_blocks": True,
        }
        for key, value in pipeline_defaults.items():
            self._pipeline_kwargs.setdefault(key, value)

        if PaddleOCRVL is not None and _supports_paddleocr_vl():
            try:
                self._pipeline = PaddleOCRVL(**self._pipeline_kwargs)
                self._use_vl = True
            except Exception:  # pragma: no cover - runtime fallback
                logger.exception(
                    "Failed to initialize PaddleOCRVL. Falling back to PaddleOCR."
                )
                self._pipeline = self._build_std_pipeline()
        elif PaddleOCRVL is not None:
            logger.warning(
                "Installed Paddle runtime does not expose fused_rms_norm_ext; "
                "disabling PaddleOCRVL and falling back to PaddleOCR."
            )
            self._pipeline = self._build_std_pipeline()
        elif PaddleOCR is not None:  # pragma: no cover - runtime fallback
            self._pipeline = self._build_std_pipeline()
        else:  # pragma: no cover - runtime fallback
            logger.warning(
                "Neither PaddleOCRVL nor PaddleOCR is installed. "
                "OCR will return empty results."
            )

    def extract(self, image_path: str) -> OCRExecutionResult:
        """이미지 경로를 입력받아 OCR 결과를 반환한다."""
        if self._pipeline is None:
            return OCRExecutionResult(tokens=[], combined_text="")
        if self._use_vl:
            try:
                ocr_outputs = self._pipeline.predict(image_path)
                return self._extract_from_vl(ocr_outputs)
            except Exception:  # pragma: no cover - runtime fallback
                logger.exception(
                    "PaddleOCRVL inference failed. Retrying with standard PaddleOCR."
                )
                self._pipeline = self._build_std_pipeline()
                self._use_vl = False
                if self._pipeline is None:
                    return OCRExecutionResult(tokens=[], combined_text="")
        return self._extract_from_std(self._run_std_ocr(image_path))

    def _build_std_pipeline(self):
        if PaddleOCR is None:  # pragma: no cover - runtime fallback
            logger.warning("PaddleOCR fallback is not installed.")
            return None

        if self._pipeline_kwargs:
            logger.warning(
                "Falling back to PaddleOCR. Ignoring PaddleOCRVL-only kwargs: %s",
                self._pipeline_kwargs,
            )

        return PaddleOCR(use_angle_cls=True, lang="en")

    def _run_std_ocr(self, image_path: str):
        try:
            return self._pipeline.ocr(image_path)
        except TypeError:
            logger.debug("PaddleOCR.ocr(image_path) failed; retrying with predict(image_path).", exc_info=True)
            return self._pipeline.predict(image_path)

    def _extract_from_vl(self, ocr_outputs) -> OCRExecutionResult:
        filtered_tokens: List[OCRToken] = []
        combined_text_parts: List[str] = []
        markdown_parts: List[str] = []
        markdown_images: List[Tuple[str, Image.Image]] = []
        structured_parts: List[str] = []
        spotting_parts: List[str] = []
        table_htmls: Dict[str, str] = {}
        block_labels: List[str] = []
        block_contents: List[str] = []

        for doc in ocr_outputs:
            doc_dict = self._to_dict(doc)
            for token in self._extract_tokens_from_vl_dict(doc_dict):
                filtered_tokens.append(token)
                combined_text_parts.append(token.text)

            markdown_payload = self._extract_markdown_payload(doc)
            if markdown_payload:
                markdown_text = str(markdown_payload.get("markdown_texts") or "").strip()
                if markdown_text:
                    markdown_parts.append(markdown_text)
                for path, image_obj in (markdown_payload.get("markdown_images") or {}).items():
                    markdown_images.append((path, image_obj))

            json_payload = self._extract_json_payload(doc)
            if json_payload:
                parsed = self._extract_structured_blocks(json_payload)
                if parsed["text"]:
                    structured_parts.append(parsed["text"])
                if parsed["spotting_text"]:
                    spotting_parts.append(parsed["spotting_text"])
                block_labels.extend(parsed["labels"])
                block_contents.extend(parsed["contents"])

            html_payload = self._extract_html_payload(doc)
            if html_payload:
                table_htmls.update(
                    {
                        str(key): str(value)
                        for key, value in html_payload.items()
                        if str(value or "").strip()
                    }
                )

        markdown_text = "\n\n".join(part for part in markdown_parts if part).strip() or None
        structured_text = "\n\n".join(part for part in structured_parts if part).strip() or None
        spotting_text = " ".join(part for part in spotting_parts if part).strip()
        combined_text = " ".join(part for part in combined_text_parts if part).strip()
        if not combined_text and block_contents:
            combined_text = " ".join(block_contents).strip()
        return OCRExecutionResult(
            tokens=filtered_tokens,
            combined_text=combined_text,
            markdown_text=markdown_text,
            markdown_images=markdown_images or None,
            structured_text=structured_text,
            spotting_text=spotting_text,
            table_htmls=table_htmls,
            block_labels=block_labels,
            block_contents=block_contents,
        )

    def _extract_tokens_from_vl_dict(self, doc_dict: dict) -> List[OCRToken]:
        tokens: List[OCRToken] = []
        pages = doc_dict.get("pages", [])
        for page in pages:
            for line in page.get("lines", []):
                text = str(line.get("text", "")).strip()
                score = float(line.get("score", 1.0))
                if not text or score < self._score_threshold:
                    continue
                box = line.get("bbox") or line.get("box") or line.get("polygon") or []
                box = [list(map(float, pt)) for pt in box]
                tokens.append(OCRToken(text=text, score=score, box=box))
        return tokens

    @staticmethod
    def _extract_markdown_payload(doc: Any) -> Optional[dict]:
        markdown = None
        if hasattr(doc, "_to_markdown"):
            try:
                markdown = doc._to_markdown(pretty=False)
            except Exception:
                logger.debug("Failed to call PaddleOCR-VL _to_markdown(pretty=False).", exc_info=True)
        if not markdown and hasattr(doc, "markdown"):
            try:
                markdown = doc.markdown
            except Exception:
                logger.debug("Failed to access PaddleOCR-VL markdown property.", exc_info=True)
        if isinstance(markdown, dict):
            return markdown
        return None

    @staticmethod
    def _extract_json_payload(doc: Any) -> Optional[dict]:
        if hasattr(doc, "json"):
            try:
                payload = doc.json
                if isinstance(payload, dict):
                    return payload.get("res") if isinstance(payload.get("res"), dict) else payload
            except Exception:
                logger.debug("Failed to access PaddleOCR-VL json payload.", exc_info=True)
        return None

    @staticmethod
    def _extract_html_payload(doc: Any) -> Dict[str, str]:
        if hasattr(doc, "html"):
            try:
                payload = doc.html
                if isinstance(payload, dict):
                    return {
                        str(key): str(value)
                        for key, value in payload.items()
                        if str(value or "").strip()
                    }
            except Exception:
                logger.debug("Failed to access PaddleOCR-VL html payload.", exc_info=True)
        return {}

    @staticmethod
    def _extract_structured_blocks(json_payload: dict) -> Dict[str, Any]:
        parsing_res_list = json_payload.get("parsing_res_list") or []
        spotting_res = json_payload.get("spotting_res") or {}
        labels: List[str] = []
        contents: List[str] = []
        text_parts: List[str] = []
        for block in parsing_res_list:
            if not isinstance(block, dict):
                continue
            label = str(block.get("block_label") or "").strip()
            content = str(block.get("block_content") or "").strip()
            if label:
                labels.append(label)
            if content:
                contents.append(content)
                text_parts.append(content)

        spotting_text = ""
        if isinstance(spotting_res, dict):
            rec_texts = spotting_res.get("rec_texts") or []
            spotting_text = " ".join(str(text).strip() for text in rec_texts if str(text).strip()).strip()
            if spotting_text:
                text_parts.append(spotting_text)

        return {
            "labels": labels,
            "contents": contents,
            "text": "\n\n".join(part for part in text_parts if part).strip(),
            "spotting_text": spotting_text,
        }

    def _extract_from_std(self, ocr_outputs) -> OCRExecutionResult:
        filtered_tokens: List[OCRToken] = []
        combined_text_parts: List[str] = []

        if not ocr_outputs:
            return OCRExecutionResult(tokens=[], combined_text="")

        if hasattr(ocr_outputs, "to_dict"):
            ocr_outputs = ocr_outputs.to_dict()

        if isinstance(ocr_outputs, dict):
            pages = ocr_outputs.get("pages")
            if isinstance(pages, list):
                return self._extract_std_pages(pages)
            results = ocr_outputs.get("results")
            if results is not None:
                ocr_outputs = results

        if not ocr_outputs:
            return OCRExecutionResult(tokens=[], combined_text="")

        # paddleocr 2.x returns [results] when given a single image path
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

    def _extract_std_pages(self, pages: List[dict]) -> OCRExecutionResult:
        filtered_tokens: List[OCRToken] = []
        combined_text_parts: List[str] = []

        for page in pages:
            for line in page.get("lines", []):
                text = str(line.get("text", "")).strip()
                score = float(line.get("score", 1.0) or 0.0)
                if not text or score < self._score_threshold:
                    continue
                box = line.get("bbox") or line.get("box") or line.get("polygon") or []
                box = [list(map(float, pt)) for pt in box] if box else []
                filtered_tokens.append(OCRToken(text=text, score=score, box=box))
                combined_text_parts.append(text)

        return OCRExecutionResult(
            tokens=filtered_tokens,
            combined_text=" ".join(combined_text_parts).strip(),
        )

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
