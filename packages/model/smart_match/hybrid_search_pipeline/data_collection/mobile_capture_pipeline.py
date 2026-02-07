"""
Data Collection Layer

Defines the mobile capture and quality control workflow for collecting
industrial equipment imagery prior to uploading into the preprocessing stack.

Pipeline:
    1. Mobile capture session metadata intake
    2. Automatic quality control script execution
    3. Upload to object storage (S3 or MinIO)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from botocore.client import BaseClient

logger = logging.getLogger(__name__)


class QCStep(Protocol):
    """Interface for quality control checks executed on captured assets."""

    def run(self, asset_path: Path) -> bool:
        ...


@dataclass
class CaptureMetadata:
    """Metadata captured alongside the mobile session."""

    session_id: str
    operator: str
    location: str
    device_model: str
    notes: str | None = None


class MobileCapturePipeline:
    """Co-ordinates capture ingestion, QC validation and object storage upload."""

    def __init__(
        self,
        qc_steps: Iterable[QCStep],
        bucket_name: str,
        storage_client: BaseClient,
    ) -> None:
        self._qc_steps = list(qc_steps)
        self._bucket_name = bucket_name
        self._storage_client = storage_client

    def process_asset(self, asset_path: Path, metadata: CaptureMetadata) -> None:
        """Validate a captured asset and upload to S3 compatible storage."""
        logger.info("Processing asset %s for session %s", asset_path, metadata.session_id)
        self._run_quality_control(asset_path)
        self._upload_asset(asset_path, metadata)

    def _run_quality_control(self, asset_path: Path) -> None:
        for step in self._qc_steps:
            if not step.run(asset_path):
                raise ValueError(f"QC failed for asset {asset_path} using {step.__class__.__name__}")

    def _upload_asset(self, asset_path: Path, metadata: CaptureMetadata) -> None:
        key = f"{metadata.session_id}/{asset_path.name}"
        logger.debug("Uploading %s to %s/%s", asset_path, self._bucket_name, key)
        self._storage_client.upload_file(str(asset_path), self._bucket_name, key)


__all__ = [
    "CaptureMetadata",
    "MobileCapturePipeline",
    "QCStep",
]
