from __future__ import annotations

import torch

from smart_match.hybrid_search_pipeline.preprocessing.pipeline import PreprocessingPipeline


class _DummyVisionEncoder:
    def encode(self, image_path: str):
        return torch.tensor([1.0, 2.0], dtype=torch.float32)


class _DummyOCREngine:
    def __init__(self):
        self.called = False

    def extract(self, image_path: str):
        self.called = True
        raise AssertionError("OCR engine should not be called when OCR is disabled.")


class _DummyTextEncoder:
    embedding_dim = 3

    def encode_document(self, text: str):
        return torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)


class _DummyMetadataNormalizer:
    def normalize(self, metadata):
        return dict(metadata)


def test_preprocessing_pipeline_skips_ocr_when_disabled():
    ocr_engine = _DummyOCREngine()
    pipeline = PreprocessingPipeline(
        vision_encoder=_DummyVisionEncoder(),
        ocr_engine=ocr_engine,
        text_encoder=_DummyTextEncoder(),
        metadata_normalizer=_DummyMetadataNormalizer(),
        captioner=None,
    )

    record = pipeline(
        "dummy.png",
        {"model_id": "M170", "maker": "Logitech"},
        enable_ocr=False,
    )

    assert ocr_engine.called is False
    assert record.ocr_tokens == []
    assert record.ocr_text == ""
    assert torch.equal(record.ocr_vector, torch.zeros(3))
    assert "model id: M170" in record.text_corpus
    assert "maker: Logitech" in record.text_corpus
