"""Internal catalog RAG service.

Indexes PDF catalog content into Milvus and serves semantic chunk retrieval.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import UploadFile
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from pypdf import PdfReader
from smart_match.hybrid_search_pipeline.preprocessing.embedding.bge_m3_encoder import BGEM3TextEncoder

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger("catalog_service")

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None


class _TTLCache:
    def __init__(self, *, ttl_seconds: int, max_items: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_items
        self._items: OrderedDict[str, Tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        now = time.time()
        item = self._items.get(key)
        if item is None:
            return None
        exp, value = item
        if now >= exp:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        self._items[key] = (now + self._ttl, value)
        self._items.move_to_end(key)
        while len(self._items) > self._max:
            self._items.popitem(last=False)


class CatalogRagService:
    """Indexes and retrieves internal catalog chunks."""

    def __init__(self) -> None:
        self._collection_name = "catalog_chunks"
        self._collection: Collection | None = None
        self._text_encoder: BGEM3TextEncoder | None = None
        self._ocr_engine: Any | None = None
        self._chunk_size = 900
        self._chunk_overlap = 150
        self._query_cache = _TTLCache(ttl_seconds=90, max_items=256)
        self._embed_cache = _TTLCache(ttl_seconds=300, max_items=512)

    @property
    def text_encoder(self) -> BGEM3TextEncoder:
        if self._text_encoder is None:
            logger.info("Initializing catalog text encoder...")
            self._text_encoder = BGEM3TextEncoder()
        return self._text_encoder

    @property
    def ocr_engine(self):
        if self._ocr_engine is not None:
            return self._ocr_engine
        if PaddleOCR is None:
            self._ocr_engine = False
            return None
        try:
            self._ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception:
            logger.exception("Failed to initialize PaddleOCR for catalog OCR fallback.")
            self._ocr_engine = False
        return self._ocr_engine if self._ocr_engine is not False else None

    @staticmethod
    def _escape_expr(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)?", (text or "").lower())

    @staticmethod
    def _extract_spec_tokens(text: str) -> List[str]:
        return re.findall(
            r"\d+(?:\.\d+)?(?:v|a|w|hz|mah|vac|vdc|amp|amps|volt|volts)",
            (text or "").lower().replace(" ", ""),
        )

    def _lexical_score(self, query_text: str, doc_text: str) -> float:
        q_tokens = self._tokenize_text(query_text)
        if not q_tokens:
            return 0.0
        d_tokens = self._tokenize_text(doc_text)
        if not d_tokens:
            return 0.0
        d_set = set(d_tokens)
        overlap = sum(1 for t in q_tokens if t in d_set)
        overlap_ratio = overlap / max(1.0, float(len(q_tokens)))
        substring_bonus = 0.2 if query_text.lower().strip() in doc_text.lower() else 0.0
        return min(1.0, overlap_ratio + substring_bonus)

    def _spec_score(self, query_text: str, doc_text: str, part_number: str) -> float:
        specs = set(self._extract_spec_tokens(query_text))
        normalized_doc = (doc_text or "").lower().replace(" ", "")
        spec_overlap = 0.0
        if specs:
            matched = sum(1 for s in specs if s in normalized_doc)
            spec_overlap = matched / max(1.0, float(len(specs)))
        pn = (part_number or "").strip().lower()
        q = (query_text or "").strip().lower()
        pn_bonus = 0.0
        if pn and q:
            if pn == q:
                pn_bonus = 1.0
            elif pn in q or q in pn:
                pn_bonus = 0.6
        return min(1.0, spec_overlap * 0.7 + pn_bonus * 0.8)

    def _extract_page_text(self, page) -> str:
        # Keep line boundaries for table-like documents.
        text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
        lines = [line.rstrip() for line in text.splitlines()]
        merged: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            merged.append(stripped)
        return "\n".join(merged).strip()

    def _extract_page_image_ocr(self, page) -> str:
        engine = self.ocr_engine
        if engine is None:
            return ""
        collected: List[str] = []
        for img in getattr(page, "images", []) or []:
            try:
                raw = getattr(img, "data", None)
                if not raw:
                    continue
                suffix = Path(str(getattr(img, "name", "image.png"))).suffix or ".png"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_img:
                    tmp_img.write(raw)
                    img_path = Path(tmp_img.name)
                try:
                    ocr_res = engine.ocr(str(img_path), cls=True)
                finally:
                    img_path.unlink(missing_ok=True)
                if not ocr_res:
                    continue
                lines = ocr_res[0] if isinstance(ocr_res[0], list) else ocr_res
                for line in lines:
                    if not line or len(line) < 2:
                        continue
                    text_info = line[1]
                    if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                        continue
                    text = str(text_info[0] or "").strip()
                    if text:
                        collected.append(text)
            except Exception:
                logger.debug("Failed image OCR fallback on a PDF image.", exc_info=True)
        return " ".join(collected).strip()

    def _ensure_collection(self) -> Collection:
        if self._collection is not None:
            return self._collection

        connections.connect(alias="default", uri=settings.MILVUS_URI)
        if not utility.has_collection(self._collection_name):
            logger.info("Creating Milvus collection: %s", self._collection_name)
            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.text_encoder.embedding_dim),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="page", dtype=DataType.INT64),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="model_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="part_number", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="maker", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=4096),
            ]
            schema = CollectionSchema(fields=fields, description="Internal PDF catalog chunks")
            collection = Collection(name=self._collection_name, schema=schema)
            collection.create_index(
                field_name="vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        else:
            collection = Collection(name=self._collection_name)

        collection.load()
        self._collection = collection
        return collection

    def _chunk_text(self, text: str) -> List[str]:
        if not (text or "").strip():
            return []
        # Preserve table-ish rows by splitting lines first.
        rows = [re.sub(r"\s+", " ", row).strip() for row in text.splitlines() if row.strip()]
        normalized = "\n".join(rows).strip()
        chunks: List[str] = []
        start = 0
        text_len = len(normalized)
        while start < text_len:
            end = min(text_len, start + self._chunk_size)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start = max(0, end - self._chunk_overlap)
        return chunks

    def index_pdf(
        self,
        *,
        pdf: UploadFile,
        source: Optional[str] = None,
        model_id: str = "",
        part_number: str = "",
        maker: str = "",
    ) -> Dict[str, object]:
        filename = (pdf.filename or "catalog.pdf").strip()
        source_name = (source or filename).strip() or "catalog.pdf"
        document_id = uuid.uuid4().hex[:12]

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf.file.read())
            pdf_path = Path(tmp.name)

        pages_indexed = 0
        chunks_indexed = 0
        try:
            reader = PdfReader(str(pdf_path))
            collection = self._ensure_collection()

            pks: List[str] = []
            vectors: List[List[float]] = []
            doc_ids: List[str] = []
            sources: List[str] = []
            pages: List[int] = []
            chunk_ids: List[int] = []
            model_ids: List[str] = []
            part_numbers: List[str] = []
            makers: List[str] = []
            chunk_texts: List[str] = []

            for page_idx, page in enumerate(reader.pages, start=1):
                text = self._extract_page_text(page)
                if not text:
                    # Scanned or image-heavy PDFs: OCR fallback on embedded images.
                    text = self._extract_page_image_ocr(page)
                chunks = self._chunk_text(text)
                if not chunks:
                    continue
                pages_indexed += 1
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    pk = f"{document_id}::p{page_idx:04d}::c{chunk_idx:04d}"
                    vector = self.text_encoder.encode_document(chunk).tolist()
                    pks.append(pk)
                    vectors.append(vector)
                    doc_ids.append(document_id)
                    sources.append(source_name)
                    pages.append(page_idx)
                    chunk_ids.append(chunk_idx)
                    model_ids.append((model_id or "").strip())
                    part_numbers.append((part_number or "").strip())
                    makers.append((maker or "").strip())
                    chunk_texts.append(chunk[:4096])
                    chunks_indexed += 1

            if pks:
                collection.insert(
                    [
                        pks,
                        vectors,
                        doc_ids,
                        sources,
                        pages,
                        chunk_ids,
                        model_ids,
                        part_numbers,
                        makers,
                        chunk_texts,
                    ]
                )
                collection.flush()
            logger.info(
                "Catalog PDF indexed: source=%s document_id=%s pages=%d chunks=%d",
                source_name,
                document_id,
                pages_indexed,
                chunks_indexed,
            )
            return {
                "status": "indexed",
                "document_id": document_id,
                "source": source_name,
                "pages_indexed": pages_indexed,
                "chunks_indexed": chunks_indexed,
            }
        finally:
            pdf_path.unlink(missing_ok=True)

    def search(
        self,
        *,
        query_text: str,
        top_k: int = 10,
        model_id: Optional[str] = None,
        part_number: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        q = (query_text or "").strip()
        if not q:
            return []
        top_k = max(1, min(50, int(top_k or 10)))
        search_k = max(top_k, min(100, top_k * 3))

        filters: List[str] = []
        if model_id and model_id.strip():
            filters.append(f'model_id == "{self._escape_expr(model_id.strip())}"')
        if part_number and part_number.strip():
            filters.append(f'part_number == "{self._escape_expr(part_number.strip())}"')
        expr = " and ".join(filters) if filters else None

        cache_key = hashlib.sha1(
            f"{q}|{top_k}|{model_id or ''}|{part_number or ''}".encode("utf-8")
        ).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        embed_key = hashlib.sha1(f"catalog_q::{q}".encode("utf-8")).hexdigest()
        cached_vector = self._embed_cache.get(embed_key)
        if cached_vector is not None:
            vector = cached_vector
        else:
            vector = self.text_encoder.encode_query(q).tolist()
            self._embed_cache.set(embed_key, vector)

        collection = self._ensure_collection()
        search_kwargs = {
            "data": [vector],
            "anns_field": "vector",
            "param": {"metric_type": "COSINE", "params": {"ef": 64}},
            "limit": search_k,
            "output_fields": [
                "document_id",
                "source",
                "page",
                "chunk_id",
                "model_id",
                "part_number",
                "maker",
                "chunk_text",
            ],
        }
        if expr:
            search_kwargs["expr"] = expr
        hits = collection.search(**search_kwargs)[0]

        results: List[Dict[str, object]] = []
        for hit in hits:
            entity = hit.entity
            base_score = max(0.0, 1.0 - float(hit.distance))
            text_value = str(entity.get("chunk_text") or "")
            lexical_score = self._lexical_score(q, text_value)
            spec_score = self._spec_score(q, text_value, str(entity.get("part_number") or ""))
            final_score = min(1.0, base_score * 0.65 + lexical_score * 0.20 + spec_score * 0.15)
            results.append(
                {
                    "score": final_score,
                    "document_id": str(entity.get("document_id") or ""),
                    "source": str(entity.get("source") or ""),
                    "page": int(entity.get("page") or 0),
                    "chunk_id": int(entity.get("chunk_id") or 0),
                    "model_id": str(entity.get("model_id") or ""),
                    "part_number": str(entity.get("part_number") or ""),
                    "maker": str(entity.get("maker") or ""),
                    "text": str(entity.get("chunk_text") or ""),
                    "lexical_score": lexical_score,
                    "spec_match_score": spec_score,
                }
            )
        results.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        final_results = results[:top_k]
        self._query_cache.set(cache_key, final_results)
        return final_results


catalog_service = CatalogRagService()
