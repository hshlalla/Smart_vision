"""
Milvus Hybrid Index Layer

Defines the three Milvus collections that store dense vectors and structured attrs:
    1. image_parts  - dense image embeddings (BGE-VL)
    2. text_parts   - dense OCR/text embeddings (BGE-M3/E5-Large)
    3. attrs_parts  - structured metadata fields (maker, part number, category)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

import logging

logger = logging.getLogger(__name__)


@dataclass
class FieldSpec:
    name: str
    dtype: DataType
    max_length: Optional[int] = None


@dataclass
class CollectionConfig:
    """Configuration for a single Milvus collection."""

    name: str
    dimension: int
    metric_type: str = "COSINE"
    index_type: str = "HNSW"
    ef_construction: int = 200
    M: int = 16
    extra_fields: Sequence[FieldSpec] = ()
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
    for field_spec in config.extra_fields:
        field_kwargs = {}
        if field_spec.dtype == DataType.VARCHAR and field_spec.max_length:
            field_kwargs["max_length"] = field_spec.max_length
        fields.append(FieldSchema(name=field_spec.name, dtype=field_spec.dtype, **field_kwargs))
    return CollectionSchema(fields, description=f"{config.name} embeddings")


class HybridMilvusIndex:
    """Manages the set of Milvus collections used in hybrid retrieval."""

    def __init__(
        self,
        image_cfg: CollectionConfig,
        text_cfg: CollectionConfig,
        attrs_fields: Sequence[FieldSpec],
        *,
        model_cfg: Optional[CollectionConfig] = None,
        caption_cfg: Optional[CollectionConfig] = None,
        image_collection_name: str | None = None,
        text_collection_name: str | None = None,
        attrs_collection_name: str = "attrs_parts",
        model_collection_name: Optional[str] = None,
        caption_collection_name: Optional[str] = None,
    ):
        self._image_cfg = image_cfg
        self._text_cfg = text_cfg
        self._model_cfg = model_cfg
        self._caption_cfg = caption_cfg
        self._attrs_fields = list(attrs_fields)
        pk_lengths = [image_cfg.pk_max_length, text_cfg.pk_max_length]
        if caption_cfg:
            pk_lengths.append(caption_cfg.pk_max_length)
        pk_lengths.append(128)
        self._pk_max_length = max(pk_lengths)
        image_name = image_collection_name or image_cfg.name
        text_name = text_collection_name or text_cfg.name
        attrs_name = attrs_collection_name
        model_name = model_collection_name or (model_cfg.name if model_cfg else None)
        caption_name = caption_collection_name or (caption_cfg.name if caption_cfg else None)

        self.image_collection = self._get_or_create_collection(
            image_name, build_collection_schema(image_cfg)
        )
        self.text_collection = self._get_or_create_collection(
            text_name, build_collection_schema(text_cfg)
        )
        self.model_collection = None
        self._model_extra_fields: Sequence[FieldSpec] = ()
        if model_cfg and model_name:
            self.model_collection = self._get_or_create_collection(model_name, build_collection_schema(model_cfg))
            self._model_extra_fields = list(model_cfg.extra_fields)
        self._attrs_vector_dim = 2
        self.attrs_collection = self._get_or_create_collection(
            attrs_name, self._create_attrs_schema(attrs_name, self._attrs_fields, self._attrs_vector_dim)
        )
        self.caption_collection = None
        if caption_cfg and caption_name:
            self.caption_collection = self._get_or_create_collection(
                caption_name, build_collection_schema(caption_cfg)
            )

    def _create_attrs_schema(
        self,
        collection_name: str,
        attrs_fields: Sequence[FieldSpec],
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
        for field_spec in attrs_fields:
            field_kwargs = {}
            if field_spec.dtype == DataType.VARCHAR:
                field_kwargs["max_length"] = field_spec.max_length or 512
            fields.append(FieldSchema(name=field_spec.name, dtype=field_spec.dtype, **field_kwargs))
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
        if self.model_collection and self._model_cfg:
            self.model_collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": self._model_cfg.index_type,
                    "metric_type": self._model_cfg.metric_type,
                    "params": {"efConstruction": self._model_cfg.ef_construction, "M": self._model_cfg.M},
                },
            )
        if self.caption_collection and self._caption_cfg:
            self.caption_collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": self._caption_cfg.index_type,
                    "metric_type": self._caption_cfg.metric_type,
                    "params": {"efConstruction": self._caption_cfg.ef_construction, "M": self._caption_cfg.M},
                },
            )

    def insert(
        self,
        primary_keys: Sequence[str],
        image_vectors: Iterable[Sequence[float]],
        text_vectors: Iterable[Sequence[float]],
        attrs_rows: List[Dict[str, object]],
        *,
        caption_vectors: Optional[Iterable[Sequence[float]]] = None,
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

        if self.caption_collection:
            if caption_vectors is None:
                raise ValueError("caption_vectors must be provided when caption collection is configured.")
            caption_data = [list(vec) for vec in caption_vectors]
            if len(caption_data) != len(primary_keys):
                raise ValueError("Primary key count must match number of caption vectors.")
            self.caption_collection.insert([primary_keys, caption_data])

        attr_vectors = [[0.0] * self._attrs_vector_dim for _ in attrs_rows]
        if len(attrs_rows) != len(primary_keys):
            raise ValueError("Primary key count must match number of attribute rows.")

        self.attrs_collection.insert(
            self._build_row_payload(
                self.attrs_collection.schema,
                primary_keys=primary_keys,
                vectors=attr_vectors,
                rows=attrs_rows,
            )
        )

    def flush(self) -> None:
        """Flush all collections to ensure data is persisted."""
        self.image_collection.flush()
        self.text_collection.flush()
        self.attrs_collection.flush()
        if self.caption_collection:
            self.caption_collection.flush()
        if self.model_collection:
            self.model_collection.flush()

    def load(self) -> None:
        """Load collections into memory for search operations."""
        self.image_collection.load()
        self.text_collection.load()
        self.attrs_collection.load()
        if self.model_collection:
            self.model_collection.load()
        if self.caption_collection:
            self.caption_collection.load()

    @staticmethod
    def _build_row_payload(
        schema: CollectionSchema,
        *,
        primary_keys: Sequence[str],
        vectors: Sequence[Sequence[float]],
        rows: Sequence[Dict[str, object]],
    ) -> List[List[object]]:
        """Build a payload aligned to the *actual* collection schema.

        This makes inserts resilient to existing collections created with older schemas.
        Unknown/missing fields are filled with empty strings / zeros as appropriate.
        """
        fields = list(schema.fields)
        field_names = [field.name for field in fields]
        missing = [name for name in ("pk", "vector") if name not in field_names]
        if missing:
            raise RuntimeError(f"Milvus schema is missing required fields: {missing}")

        payload: List[List[object]] = []
        for field in fields:
            if field.name == "pk":
                payload.append([str(pk) for pk in primary_keys])
                continue
            if field.name == "vector":
                payload.append([list(vec) for vec in vectors])
                continue

            column: List[object] = []
            for row in rows:
                value = row.get(field.name, "")
                if value is None:
                    value = ""
                if field.dtype == DataType.VARCHAR:
                    value = str(value)
                    max_len = getattr(field, "max_length", None)
                    if max_len:
                        value = value[: int(max_len)]
                column.append(value)
            payload.append(column)

        # Helpful warning when code provides fields not present in schema (old collection).
        extra_keys = set()
        for row in rows:
            extra_keys.update(row.keys())
        extras = sorted(extra_keys - set(field_names))
        if extras:
            logger.warning(
                "Milvus attrs schema has no fields %s; values will be ignored. "
                "Drop/recreate the collection to upgrade schema.",
                extras,
            )

        return payload

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

    def search_captions(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        search_params: Dict[str, object] | None = None,
        output_fields: Sequence[str] | None = None,
    ):
        if not self.caption_collection or not self._caption_cfg:
            return []
        params = search_params or {"metric_type": self._caption_cfg.metric_type, "params": {"ef": 64}}
        results = self.caption_collection.search(
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

    def query_attributes_by_model(
        self,
        model_id: str,
        output_fields: Sequence[str],
    ) -> List[Dict[str, object]]:
        """Query attribute rows that belong to the specified model."""
        if not model_id:
            return []
        safe_model_id = str(model_id).replace('"', '\\"')
        expr = f'model_id == "{safe_model_id}"'
        try:
            return self.attrs_collection.query(expr=expr, output_fields=list(output_fields))
        except Exception:
            return []

    def upsert_model(self, *, model_id: str, text_vector: Sequence[float], extra_values: Dict[str, object]) -> None:
        """Insert or replace model-level text embeddings."""
        if not self.model_collection:
            raise RuntimeError("Model collection is not configured.")
        pk = str(model_id)
        expr = f'pk in ["{pk}"]'
        try:
            self.model_collection.delete(expr)
        except Exception:
            # Ignore delete failures; the insert below will still succeed.
            pass

        columns = [
            [pk],
            [list(text_vector)],
        ]

        for field_spec in self._model_extra_fields:
            value = extra_values.get(field_spec.name, "")
            if value is None:
                value = ""
            if field_spec.dtype == DataType.VARCHAR:
                value = str(value)
                if field_spec.max_length:
                    value = value[: field_spec.max_length]
            columns.append([value])

        self.model_collection.insert(columns)

    def get_model(self, model_id: str, output_fields: Optional[Sequence[str]] = None) -> Optional[Dict[str, object]]:
        if not self.model_collection:
            return None
        fields = list(output_fields) if output_fields else ["*"]
        expr = f'pk in ["{str(model_id)}"]'
        rows = self.model_collection.query(expr=expr, output_fields=fields)
        return rows[0] if rows else None

    def fetch_models(self, model_ids: Sequence[str], output_fields: Sequence[str]) -> Dict[str, Dict[str, object]]:
        if not model_ids or not self.model_collection:
            return {}
        ids = [str(mid) for mid in model_ids]
        quoted = ",".join(f'"{mid}"' for mid in ids)
        expr = f"pk in [{quoted}]"
        rows = self.model_collection.query(expr=expr, output_fields=list(output_fields))
        return {row["pk"]: row for row in rows}

    def search_models(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        search_params: Optional[Dict[str, object]] = None,
        output_fields: Sequence[str] | None = None,
    ):
        if not self.model_collection or not self._model_cfg:
            return []
        params = search_params or {
            "metric_type": self._model_cfg.metric_type,
            "params": {"ef": 64},
        }
        results = self.model_collection.search(
            data=[list(query_vector)],
            anns_field="vector",
            param=params,
            limit=top_k,
            output_fields=output_fields,
        )
        return results[0]

    def describe(self) -> Dict[str, Dict[str, object]]:
        """Return information about managed collections."""
        info: Dict[str, Dict[str, object]] = {}
        collections = [
            ("image", self.image_collection),
            ("text", self.text_collection),
            ("attrs", self.attrs_collection),
        ]
        if self.model_collection:
            collections.append(("model", self.model_collection))
        if self.caption_collection:
            collections.append(("caption", self.caption_collection))

        for label, collection in collections:
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


__all__ = ["CollectionConfig", "FieldSpec", "HybridMilvusIndex"]
