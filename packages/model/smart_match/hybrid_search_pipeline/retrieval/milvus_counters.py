"""
Milvus Counter Store

Provides a simple per-key counter stored in a dedicated Milvus collection.
This is intended for single-writer deployments (one API instance).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility


@dataclass(frozen=True)
class CounterValue:
    key: str
    value: int
    updated_at: int


class MilvusCounterStore:
    def __init__(self, *, collection_name: str = "sv_counters") -> None:
        self.collection_name = (collection_name or "sv_counters").strip()
        if not self.collection_name:
            raise ValueError("collection_name must be non-empty.")
        self.collection = self._get_or_create(self.collection_name)

    @staticmethod
    def _schema(name: str) -> CollectionSchema:
        fields = [
            FieldSchema(
                name="key",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=256,
            ),
            FieldSchema(name="value", dtype=DataType.INT64),
            FieldSchema(name="updated_at", dtype=DataType.INT64),
            # Milvus 2.6+ may reject schemas without any vector field.
            # Keep a tiny placeholder vector to satisfy schema checks.
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2),
        ]
        return CollectionSchema(fields, description=f"{name} counters")

    def _get_or_create(self, name: str) -> Collection:
        if utility.has_collection(name):
            collection = Collection(name=name)
            # Validate expected fields exist (best-effort).
            field_names = {field.name for field in collection.schema.fields}
            if not {"key", "value", "updated_at"}.issubset(field_names):
                raise RuntimeError(
                    f"Milvus counters collection '{name}' has incompatible schema. "
                    "Drop it so it can be recreated."
                )
            return collection
        return Collection(name=name, schema=self._schema(name))

    @staticmethod
    def _build_insert_payload(collection: Collection, key: str, value: int, updated_at: int):
        """Build insert payload aligned to existing collection schema."""
        payload = []
        for field in collection.schema.fields:
            if field.name == "key":
                payload.append([key])
            elif field.name == "value":
                payload.append([value])
            elif field.name == "updated_at":
                payload.append([updated_at])
            elif field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
                payload.append([[0.0, 0.0]])
            else:
                # Fallback for unexpected optional fields.
                payload.append([""])
        return payload

    def get(self, key: str) -> Optional[CounterValue]:
        key = (key or "").strip()
        if not key:
            return None
        safe = key.replace('"', '\\"')
        rows = self.collection.query(expr=f'key == "{safe}"', output_fields=["key", "value", "updated_at"])
        if not rows:
            return None
        row = rows[0]
        return CounterValue(
            key=str(row.get("key", "")),
            value=int(row.get("value", 0)),
            updated_at=int(row.get("updated_at", 0)),
        )

    def set(self, key: str, value: int) -> CounterValue:
        key = (key or "").strip()
        if not key:
            raise ValueError("key must be non-empty.")
        value = int(value)
        updated_at = int(time.time())
        safe = key.replace('"', '\\"')
        try:
            self.collection.delete(expr=f'key == "{safe}"')
        except Exception:
            pass
        self.collection.insert(self._build_insert_payload(self.collection, key, value, updated_at))
        try:
            self.collection.flush()
        except Exception:
            pass
        return CounterValue(key=key, value=value, updated_at=updated_at)

    def next(self, key: str) -> CounterValue:
        current = self.get(key)
        next_value = 1 if current is None else int(current.value) + 1
        return self.set(key, next_value)


__all__ = ["CounterValue", "MilvusCounterStore"]
