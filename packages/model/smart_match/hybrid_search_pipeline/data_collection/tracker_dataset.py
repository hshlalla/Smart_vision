"""
Tracker Dataset Loader

Utility for loading equipment metadata from ``tracker_subset.csv`` and
exposing convenient lookups by ``MODEL_ID``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackerRecord:
    """Normalized metadata row extracted from the tracker dataset."""

    category_code: str
    std_maker_name: str
    model_id: str
    non_std_model_name: str
    std_model_name: str
    model_name: str
    raw: Dict[str, str]


class TrackerDataset:
    """Lazily loads tracker metadata and provides lookup helpers."""

    REQUIRED_COLUMNS = {
        "Category_Code",
        "STD_MAKER_NAME",
        "MODEL_ID",
        "NON_STD_MODEL_NAME",
        "STD_MODEL_NAME",
    }

    def __init__(self, rows: Dict[str, TrackerRecord]) -> None:
        self._rows = rows

    @staticmethod
    def _compose_model_name(non_std: str, std: str) -> str:
        non_std = (non_std or "").strip()
        std = (std or "").strip()
        if non_std and std and non_std != std:
            return f"{non_std} / {std}"
        return non_std or std

    @classmethod
    def from_csv(cls, path: Path | str, *, encoding: str = "utf-8") -> "TrackerDataset":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tracker dataset not found: {path}")

        with path.open(encoding=encoding, errors="ignore", newline="") as fp:
            reader = csv.reader(fp)
            try:
                headers = next(reader)
            except StopIteration as exc:  # pragma: no cover - empty CSV guard
                raise ValueError(f"Tracker dataset is empty: {path}") from exc

            header_map = {name.strip(): idx for idx, name in enumerate(headers) if name.strip()}
            missing = cls.REQUIRED_COLUMNS - header_map.keys()
            if missing:
                raise ValueError(f"Tracker dataset missing required columns: {sorted(missing)}")

            records: Dict[str, TrackerRecord] = {}
            for row_index, row in enumerate(reader, start=2):
                try:
                    model_id = row[header_map["MODEL_ID"]].strip()
                except IndexError:
                    logger.debug("Skipping row %d: missing MODEL_ID", row_index)
                    continue

                if not model_id:
                    continue

                category_code = row[header_map["Category_Code"]].strip() if len(row) > header_map["Category_Code"] else ""
                maker_name = row[header_map["STD_MAKER_NAME"]].strip() if len(row) > header_map["STD_MAKER_NAME"] else ""
                non_std_name = row[header_map["NON_STD_MODEL_NAME"]].strip() if len(row) > header_map["NON_STD_MODEL_NAME"] else ""
                std_name = row[header_map["STD_MODEL_NAME"]].strip() if len(row) > header_map["STD_MODEL_NAME"] else ""
                model_name = cls._compose_model_name(non_std_name, std_name)

                raw_row = {header: row[idx].strip() if idx < len(row) else "" for header, idx in header_map.items()}
                records[model_id] = TrackerRecord(
                    category_code=category_code,
                    std_maker_name=maker_name,
                    model_id=model_id,
                    non_std_model_name=non_std_name,
                    std_model_name=std_name,
                    model_name=model_name,
                    raw=raw_row,
                )

        logger.info("Loaded %d tracker metadata rows from %s", len(records), path)
        return cls(records)

    def get(self, model_id: str) -> Optional[TrackerRecord]:
        return self._rows.get(model_id)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._rows if isinstance(model_id, str) else False

    def keys(self) -> Iterable[str]:  # pragma: no cover - simple proxy
        return self._rows.keys()

    def values(self) -> Iterable[TrackerRecord]:  # pragma: no cover - simple proxy
        return self._rows.values()


__all__ = ["TrackerDataset", "TrackerRecord"]
