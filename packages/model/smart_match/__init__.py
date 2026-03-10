"""
Smart Match package exposing the hybrid search pipeline.

The modules are organized under ``smart_match.hybrid_search_pipeline`` so
callers can import the orchestrator and supporting utilities directly from
this namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hybrid_search_pipeline.hybrid_pipeline_runner import (
        FusionWeights,
        HybridSearchOrchestrator,
        MilvusConnectionConfig,
    )
    from .hybrid_search_pipeline.preprocessing.pipeline import PreprocessingPipeline

__all__ = [
    "HybridSearchOrchestrator",
    "MilvusConnectionConfig",
    "FusionWeights",
    "PreprocessingPipeline",
]


def __getattr__(name: str) -> Any:
    if name in {"HybridSearchOrchestrator", "MilvusConnectionConfig", "FusionWeights"}:
        from .hybrid_search_pipeline.hybrid_pipeline_runner import (
            FusionWeights,
            HybridSearchOrchestrator,
            MilvusConnectionConfig,
        )

        exports = {
            "HybridSearchOrchestrator": HybridSearchOrchestrator,
            "MilvusConnectionConfig": MilvusConnectionConfig,
            "FusionWeights": FusionWeights,
        }
        return exports[name]

    if name == "PreprocessingPipeline":
        from .hybrid_search_pipeline.preprocessing.pipeline import PreprocessingPipeline

        return PreprocessingPipeline

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
