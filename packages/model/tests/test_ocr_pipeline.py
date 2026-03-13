from __future__ import annotations

from smart_match.hybrid_search_pipeline.preprocessing.ocr.OCR import PaddleOCRVLPipeline


def test_extract_from_std_supports_legacy_output():
    pipeline = PaddleOCRVLPipeline.__new__(PaddleOCRVLPipeline)
    pipeline._score_threshold = 0.5

    outputs = [
        [
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("홍수훈", 0.99)],
            [[[0, 0], [1, 0], [1, 1], [0, 1]], ("ignored", 0.2)],
        ]
    ]

    result = pipeline._extract_from_std(outputs)
    assert result.combined_text == "홍수훈"
    assert len(result.tokens) == 1


def test_extract_from_std_supports_v3_page_dict():
    pipeline = PaddleOCRVLPipeline.__new__(PaddleOCRVLPipeline)
    pipeline._score_threshold = 0.5

    outputs = {
        "pages": [
            {
                "lines": [
                    {"text": "Logitech", "score": 0.97, "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                    {"text": "ignored", "score": 0.1, "bbox": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                ]
            }
        ]
    }

    result = pipeline._extract_from_std(outputs)
    assert result.combined_text == "Logitech"
    assert len(result.tokens) == 1
