from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def _ensure_torch_stub() -> None:
    if importlib.util.find_spec("torch") is not None:
        return
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_mod


def _ensure_smart_match_stub() -> None:
    if importlib.util.find_spec("smart_match") is not None:
        return
    smart_match = types.ModuleType("smart_match")

    class HybridSearchOrchestrator:
        def __init__(self, *args, **kwargs):
            pass

        def preprocess_and_index(self, *args, **kwargs):
            return None

        def search(self, *args, **kwargs):
            return []

        def index_model_metadata(self, *args, **kwargs):
            return {}

        def allocate_model_id(self, *args, **kwargs):
            return "a000001"

    class FusionWeights:
        def __init__(self, alpha: float, beta: float, gamma: float):
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

    class MilvusConnectionConfig:
        def __init__(self, uri: str):
            self.uri = uri

    class BGEM3TextEncoder:
        embedding_dim = 2

        def encode_document(self, *_args, **_kwargs):
            class _Vec:
                def tolist(self):
                    return [0.0, 0.0]

            return _Vec()

        def encode_query(self, *_args, **_kwargs):
            class _Vec:
                def tolist(self):
                    return [0.0, 0.0]

            return _Vec()

    smart_match.HybridSearchOrchestrator = HybridSearchOrchestrator
    smart_match.FusionWeights = FusionWeights
    smart_match.MilvusConnectionConfig = MilvusConnectionConfig

    hspr = types.ModuleType("smart_match.hybrid_search_pipeline.hybrid_pipeline_runner")
    hspr.HybridSearchOrchestrator = HybridSearchOrchestrator
    hspr.FusionWeights = FusionWeights
    hspr.MilvusConnectionConfig = MilvusConnectionConfig

    enc = types.ModuleType("smart_match.hybrid_search_pipeline.preprocessing.embedding.bge_m3_encoder")
    enc.BGEM3TextEncoder = BGEM3TextEncoder

    sys.modules["smart_match"] = smart_match
    sys.modules["smart_match.hybrid_search_pipeline"] = types.ModuleType("smart_match.hybrid_search_pipeline")
    sys.modules["smart_match.hybrid_search_pipeline.hybrid_pipeline_runner"] = hspr
    sys.modules["smart_match.hybrid_search_pipeline.preprocessing"] = types.ModuleType(
        "smart_match.hybrid_search_pipeline.preprocessing"
    )
    sys.modules["smart_match.hybrid_search_pipeline.preprocessing.embedding"] = types.ModuleType(
        "smart_match.hybrid_search_pipeline.preprocessing.embedding"
    )
    sys.modules["smart_match.hybrid_search_pipeline.preprocessing.embedding.bge_m3_encoder"] = enc


def _ensure_pymilvus_stub() -> None:
    if importlib.util.find_spec("pymilvus") is not None:
        return
    pymilvus = types.ModuleType("pymilvus")

    class DataType:
        VARCHAR = "VARCHAR"
        INT64 = "INT64"
        FLOAT_VECTOR = "FLOAT_VECTOR"
        BINARY_VECTOR = "BINARY_VECTOR"

    class FieldSchema:
        def __init__(self, name: str, dtype: str, **kwargs):
            self.name = name
            self.dtype = dtype
            for k, v in kwargs.items():
                setattr(self, k, v)

    class CollectionSchema:
        def __init__(self, fields, description: str = ""):
            self.fields = fields
            self.description = description

    class _Hit:
        def __init__(self, entity=None, distance: float = 1.0):
            self.entity = entity or {}
            self.distance = distance

    class Collection:
        def __init__(self, name: str, schema: CollectionSchema | None = None):
            self.name = name
            self.schema = schema or CollectionSchema([])

        def create_index(self, *args, **kwargs):
            return None

        def load(self):
            return None

        def insert(self, *args, **kwargs):
            return None

        def flush(self):
            return None

        def query(self, *args, **kwargs):
            return []

        def delete(self, *args, **kwargs):
            return None

        def search(self, *args, **kwargs):
            return [[_Hit()]]

    pymilvus.Collection = Collection
    pymilvus.CollectionSchema = CollectionSchema
    pymilvus.DataType = DataType
    pymilvus.FieldSchema = FieldSchema
    pymilvus.connections = types.SimpleNamespace(connect=lambda **_kwargs: None)
    pymilvus.utility = types.SimpleNamespace(has_collection=lambda _name: False)
    sys.modules["pymilvus"] = pymilvus


def _ensure_pypdf_stub() -> None:
    if importlib.util.find_spec("pypdf") is not None:
        return
    pypdf = types.ModuleType("pypdf")

    class PdfReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = []

    pypdf.PdfReader = PdfReader
    sys.modules["pypdf"] = pypdf


_ensure_torch_stub()
_ensure_smart_match_stub()
_ensure_pymilvus_stub()
_ensure_pypdf_stub()
