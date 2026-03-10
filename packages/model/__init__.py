"""
Smart Match Model Package

Provides access to the Smart Vision hybrid search pipeline modules reusable
across API and demo services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
        HybridSearchOrchestrator,
    )
    from .smart_match.hybrid_search_pipeline.preprocessing.pipeline import (
        PreprocessingPipeline,
    )

__all__ = [
    "HybridSearchOrchestrator",
    "PreprocessingPipeline",
]


def __getattr__(name: str) -> Any:
    if name == "HybridSearchOrchestrator":
        from .smart_match.hybrid_search_pipeline.hybrid_pipeline_runner import (
            HybridSearchOrchestrator,
        )

        return HybridSearchOrchestrator

    if name == "PreprocessingPipeline":
        from .smart_match.hybrid_search_pipeline.preprocessing.pipeline import (
            PreprocessingPipeline,
        )

        return PreprocessingPipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
