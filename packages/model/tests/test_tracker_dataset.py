from __future__ import annotations

from smart_match.hybrid_search_pipeline.data_collection.tracker_dataset import (
    TrackerDataset,
)


def test_tracker_dataset_load_and_lookup(tmp_path):
    csv_path = tmp_path / "tracker.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Category_Code,STD_MAKER_NAME,MODEL_ID,NON_STD_MODEL_NAME,STD_MODEL_NAME",
                "ETCH,Samsung,A0001,ABC-123,ABC123",
                "CVD,Applied,A0002,,Centura",
            ]
        ),
        encoding="utf-8",
    )

    ds = TrackerDataset.from_csv(csv_path)

    row1 = ds.get("A0001")
    assert row1 is not None
    assert row1.model_name == "ABC-123 / ABC123"
    assert "A0001" in ds

    row2 = ds.get("A0002")
    assert row2 is not None
    assert row2.model_name == "Centura"


def test_tracker_dataset_validates_required_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("MODEL_ID,STD_MAKER_NAME\nA1,Samsung\n", encoding="utf-8")

    try:
        TrackerDataset.from_csv(bad_csv)
        assert False, "Expected ValueError for missing required columns"
    except ValueError as exc:
        assert "missing required columns" in str(exc).lower()
