"""
Milvus Hybrid Index Layer

Defines the three Milvus collections that store dense vectors and structured attrs:
    1. image_parts  - dense image embeddings (BGE-VL)
    2. text_parts   - dense OCR/text embeddings (BGE-M3/E5-Large)
    3. attrs_parts  - structured metadata fields (maker, part number, category)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility


@dataclass
class CollectionConfig:
    """Configuration for a single Milvus collection."""

    name: str
    dimension: int
    metric_type: str = "COSINE"
    index_type: str = "HNSW"
    ef_construction: int = 200
    M: int = 16
    pk_max_length: int = 256


def build_collection_schema(config: CollectionConfig) -> CollectionSchema:
    """Create a standard vector+id schema for Milvus collections."""
    fields = [
        FieldSchema(
            name="pk",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=config.pk_max_length,
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.dimension),
    ]
    return CollectionSchema(fields, description=f"{config.name} embeddings")


class HybridMilvusIndex:
    """Manages the set of Milvus collections used in hybrid retrieval."""

    def __init__(
        self,
        image_cfg: CollectionConfig,
        text_cfg: CollectionConfig,
        attrs_fields: Sequence[Tuple[str, DataType]],
        *,
        image_collection_name: str | None = None,
        text_collection_name: str | None = None,
        attrs_collection_name: str = "attrs_parts",
    ):
        self._image_cfg = image_cfg
        self._text_cfg = text_cfg
        self._attrs_fields = list(attrs_fields)
        self._pk_max_length = max(image_cfg.pk_max_length, text_cfg.pk_max_length, 128)
        image_name = image_collection_name or image_cfg.name
        text_name = text_collection_name or text_cfg.name
        attrs_name = attrs_collection_name

        self.image_collection = self._get_or_create_collection(
            image_name, build_collection_schema(image_cfg)
        )
        self.text_collection = self._get_or_create_collection(
            text_name, build_collection_schema(text_cfg)
        )
        self._attrs_vector_dim = 2
        self.attrs_collection = self._get_or_create_collection(
            attrs_name, self._create_attrs_schema(attrs_name, self._attrs_fields, self._attrs_vector_dim)
        )

    def _create_attrs_schema(
        self,
        collection_name: str,
        attrs_fields: Sequence[Tuple[str, DataType]],
        vector_dim: int = 2,
    ) -> CollectionSchema:
        fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=self._pk_max_length,
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        ]
        for name, dtype in attrs_fields:
            field_kwargs = {}
            if dtype == DataType.VARCHAR:
                field_kwargs["max_length"] = 512
            fields.append(FieldSchema(name=name, dtype=dtype, **field_kwargs))
        return CollectionSchema(fields, description="Structured equipment attributes")

    def _get_or_create_collection(self, name: str, schema: CollectionSchema) -> Collection:
        if utility.has_collection(name):
            collection = Collection(name=name)
            pk_fields = [field for field in collection.schema.fields if field.is_primary]
            if not pk_fields or pk_fields[0].dtype != DataType.VARCHAR:
                raise RuntimeError(
                    f"Milvus collection '{name}' is using an incompatible primary key schema. "
                    "Drop the existing collection so it can be recreated with VARCHAR primary keys."
                )
            has_vector_field = any(
                field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR) for field in collection.schema.fields
            )
            if not has_vector_field:
                raise RuntimeError(
                    f"Milvus collection '{name}' lacks a vector field. "
                    "Drop the existing collection or upgrade its schema before continuing."
                )
            return collection
        return Collection(name=name, schema=schema)

    def create_indexes(self) -> None:
        """Create vector indexes for the embeddings collections."""
        self.image_collection.create_index(
            field_name="vector",
            index_params={
                "index_type": self._image_cfg.index_type,
                "metric_type": self._image_cfg.metric_type,
                "params": {"efConstruction": self._image_cfg.ef_construction, "M": self._image_cfg.M},
            },
        )
        self.text_collection.create_index(
            field_name="vector",
            index_params={
                "index_type": self._text_cfg.index_type,
                "metric_type": self._text_cfg.metric_type,
                "params": {"efConstruction": self._text_cfg.ef_construction, "M": self._text_cfg.M},
            },
        )
        self.attrs_collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "FLAT",
                "metric_type": "L2",
            },
        )

    def insert(
        self,
        primary_keys: Sequence[str],
        image_vectors: Iterable[Sequence[float]],
        text_vectors: Iterable[Sequence[float]],
        attrs_rows: List[Dict[str, object]],
    ) -> None:
        """Insert preprocessed data into the Milvus collections."""
        primary_keys = [str(pk) for pk in primary_keys]
        if not primary_keys:
            raise ValueError("Primary keys are required for insertion.")

        image_data = [list(vec) for vec in image_vectors]
        text_data = [list(vec) for vec in text_vectors]
        if len(image_data) != len(primary_keys):
            raise ValueError("Primary key count must match number of image vectors.")
        if len(text_data) != len(primary_keys):
            raise ValueError("Primary key count must match number of text vectors.")

        self.image_collection.insert([primary_keys, image_data])
        self.text_collection.insert([primary_keys, text_data])

        attr_vectors = [[0.0] * self._attrs_vector_dim for _ in attrs_rows]
        if len(attrs_rows) != len(primary_keys):
            raise ValueError("Primary key count must match number of attribute rows.")

        insert_payload = [primary_keys, attr_vectors]
        for field_name, _ in self._attrs_fields:
            column = [row.get(field_name) for row in attrs_rows]
            insert_payload.append(column)
        self.attrs_collection.insert(insert_payload)

    def flush(self) -> None:
        """Flush all collections to ensure data is persisted."""
        self.image_collection.flush()
        self.text_collection.flush()
        self.attrs_collection.flush()

    def load(self) -> None:
        """Load collections into memory for search operations."""
        self.image_collection.load()
        self.text_collection.load()
        self.attrs_collection.load()

    def search_images(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        search_params: Dict[str, object] | None = None,
        output_fields: Sequence[str] | None = None,
    ):
        params = search_params or {"metric_type": self._image_cfg.metric_type, "params": {"ef": 64}}
        results = self.image_collection.search(
            data=[list(query_vector)],
            anns_field="vector",
            param=params,
            limit=top_k,
            output_fields=output_fields,
        )
        return results[0]

    def search_texts(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        search_params: Dict[str, object] | None = None,
        output_fields: Sequence[str] | None = None,
    ):
        params = search_params or {"metric_type": self._text_cfg.metric_type, "params": {"ef": 64}}
        results = self.text_collection.search(
            data=[list(query_vector)],
            anns_field="vector",
            param=params,
            limit=top_k,
            output_fields=output_fields,
        )
        return results[0]

    def fetch_attributes(self, primary_keys: Sequence[str], output_fields: Sequence[str]) -> List[Dict[str, object]]:
        """Fetch attribute rows for the specified primary keys."""
        if not primary_keys:
            return []
        ids = [str(pk) for pk in primary_keys]
        quoted = ",".join(f'"{pk}"' for pk in ids)
        expr = f"pk in [{quoted}]"
        return self.attrs_collection.query(expr=expr, output_fields=list(output_fields))

    def describe(self) -> Dict[str, Dict[str, object]]:
        """Return information about managed collections."""
        info: Dict[str, Dict[str, object]] = {}
        for label, collection in (
            ("image", self.image_collection),
            ("text", self.text_collection),
            ("attrs", self.attrs_collection),
        ):
            try:
                num_entities = collection.num_entities
            except Exception:  # pragma: no cover - runtime safeguard
                num_entities = None
            info[label] = {
                "name": collection.name,
                "num_entities": num_entities,
            }
        return info

    def drop_collection(self, collection_name: str) -> str:
        """Drop the specified Milvus collection and return its name."""
        collection_name = (collection_name or "").strip()
        if not collection_name:
            raise ValueError("collection_name must be provided.")
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        utility.drop_collection(collection_name)
        return collection_name


__all__ = ["CollectionConfig", "HybridMilvusIndex"]
