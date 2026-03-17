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


def test_extract_from_vl_uses_markdown_and_structured_payloads():
    class FakeDoc:
        @property
        def markdown(self):
            return {
                "markdown_texts": "| model | value |\n| --- | --- |\n| PN | ABC-123 |",
                "markdown_images": {},
            }

        @property
        def json(self):
            return {
                "res": {
                    "parsing_res_list": [
                        {"block_label": "table", "block_content": "PN\tABC-123"},
                        {"block_label": "text", "block_content": "Maker Hyundai"},
                    ],
                    "spotting_res": {"rec_texts": ["ABC-123", "12V"]},
                }
            }

        @property
        def html(self):
            return {"table_1": "<table><tr><td>PN</td><td>ABC-123</td></tr></table>"}

        def to_dict(self):
            return {"pages": []}

    pipeline = PaddleOCRVLPipeline.__new__(PaddleOCRVLPipeline)
    pipeline._score_threshold = 0.5
    pipeline._pipeline = None

    result = pipeline._extract_from_vl([FakeDoc()])

    assert "model" in (result.markdown_text or "")
    assert "Maker Hyundai" in (result.structured_text or "")
    assert result.spotting_text == "ABC-123 12V"
    assert result.table_htmls["table_1"].startswith("<table>")
    assert result.block_labels == ["table", "text"]
