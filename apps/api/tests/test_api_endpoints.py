from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from smart_vision_api.api.v1 import auth as auth_api
from smart_vision_api.api.v1 import catalog as catalog_api
from smart_vision_api.api.v1 import hybrid as hybrid_api
from smart_vision_api.core import auth as auth_core
from smart_vision_api.core.auth import require_user
from smart_vision_api.core.config import settings
from smart_vision_api.services.hybrid import HybridSearchService


def test_hybrid_search_success(monkeypatch):
    def fake_search(**kwargs):
        return [{"model_id": "a000001", "score": 0.92}]

    monkeypatch.setattr(hybrid_api.hybrid_service, "search", fake_search)

    app = FastAPI()
    app.include_router(hybrid_api.router, prefix="/api/v1")
    app.dependency_overrides[require_user] = lambda: "tester"
    client = TestClient(app)

    resp = client.post(
        "/api/v1/hybrid/search",
        json={"query_text": "etch", "image_base64": None, "top_k": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["results"][0]["model_id"] == "a000001"


def test_hybrid_index_preview_success(monkeypatch):
    def fake_preview(*, image_b64_list: list[str], metadata_mode: str | None = None, label_image_b64_list: list[str] | None = None):
        assert image_b64_list == ["abc123"]
        assert metadata_mode == "auto"
        assert label_image_b64_list == []
        return {
            "status": "preview_ready",
            "ocr_image_indices": [0],
            "label_ocr_text": "",
            "draft": {
                "model_id": "",
                "maker": "Fuji Electric",
                "part_number": "SC50BAA",
                "category": "magnetic_contactor",
                "description": "Fuji Electric magnetic contactor",
                "product_info": "magnetic contactor",
                "price_value": 120000,
                "source": "openai",
            },
        }

    monkeypatch.setattr(hybrid_api.hybrid_service, "preview_index_asset", fake_preview)

    app = FastAPI()
    app.include_router(hybrid_api.router, prefix="/api/v1")
    app.dependency_overrides[require_user] = lambda: "tester"
    client = TestClient(app)

    resp = client.post("/api/v1/hybrid/index/preview", json={"image_base64": "abc123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "preview_ready"
    assert body["draft"]["maker"] == "Fuji Electric"


def test_hybrid_index_confirm_success(monkeypatch):
    def fake_confirm(*, image_b64_list: list[str], metadata: dict):
        assert image_b64_list == ["abc123"]
        assert metadata["maker"] == "Fuji Electric"
        assert metadata["ocr_image_indices"] == [0]
        return {"status": "queued", "model_id": "m000001", "task_id": "task-1"}

    monkeypatch.setattr(hybrid_api.hybrid_service, "confirm_index_asset", fake_confirm)

    app = FastAPI()
    app.include_router(hybrid_api.router, prefix="/api/v1")
    app.dependency_overrides[require_user] = lambda: "tester"
    client = TestClient(app)

    resp = client.post(
        "/api/v1/hybrid/index/confirm",
        json={
            "image_base64": "abc123",
            "model_id": "",
            "maker": "Fuji Electric",
            "part_number": "SC50BAA",
            "category": "magnetic_contactor",
            "description": "Fuji Electric magnetic contactor",
            "product_info": "magnetic contactor",
            "price_value": 120000,
            "ocr_image_indices": [0],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "m000001"
    assert resp.json()["status"] == "queued"
    assert resp.json()["task_id"] == "task-1"


def test_hybrid_index_task_success(monkeypatch):
    def fake_get_task(task_id: str):
        assert task_id == "task-1"
        return {
            "task_id": "task-1",
            "status": "running",
            "model_id": "m000001",
            "detail": "Running OCR, embeddings, and Milvus upsert.",
        }

    monkeypatch.setattr(hybrid_api.hybrid_service, "get_index_task", fake_get_task)

    app = FastAPI()
    app.include_router(hybrid_api.router, prefix="/api/v1")
    app.dependency_overrides[require_user] = lambda: "tester"
    client = TestClient(app)

    resp = client.get("/api/v1/hybrid/index/tasks/task-1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    assert resp.json()["model_id"] == "m000001"


def test_hybrid_bulk_zip_success(monkeypatch):
    def fake_bulk_index(archive):
        assert archive.filename == "items.zip"
        return {"status": "queued", "task_id": "bulk-task-1", "job_type": "bulk_zip"}

    monkeypatch.setattr(hybrid_api.hybrid_service, "index_bulk_zip_archive", fake_bulk_index)

    app = FastAPI()
    app.include_router(hybrid_api.router, prefix="/api/v1")
    app.dependency_overrides[require_user] = lambda: "tester"
    client = TestClient(app)

    resp = client.post(
        "/api/v1/hybrid/index/bulk_zip",
        files={"archive": ("items.zip", b"fake-zip", "application/zip")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"
    assert body["task_id"] == "bulk-task-1"
    assert body["job_type"] == "bulk_zip"


def test_hybrid_search_rejects_too_large_image(monkeypatch):
    monkeypatch.setattr(settings, "MAX_IMAGE_BASE64_LENGTH", 20)

    app = FastAPI()
    app.include_router(hybrid_api.router, prefix="/api/v1")
    app.dependency_overrides[require_user] = lambda: "tester"
    client = TestClient(app)

    resp = client.post(
        "/api/v1/hybrid/search",
        json={"query_text": "etch", "image_base64": "a" * 21, "top_k": 5},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]


def test_catalog_search_success(monkeypatch):
    def fake_search(**kwargs):
        return [
            {
                "score": 0.88,
                "lexical_score": 0.5,
                "spec_match_score": 0.2,
                "document_id": "doc1",
                "source": "manual.pdf",
                "page": 3,
                "chunk_id": 1,
                "model_id": "a000001",
                "part_number": "PN-1",
                "maker": "maker",
                "text": "chunk text",
            }
        ]

    monkeypatch.setattr(catalog_api.catalog_service, "search", fake_search)

    app = FastAPI()
    app.include_router(catalog_api.router, prefix="/api/v1")
    app.dependency_overrides[require_user] = lambda: "tester"
    client = TestClient(app)

    resp = client.post("/api/v1/catalog/search", json={"query_text": "spec", "top_k": 3})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["source"] == "manual.pdf"


def test_auth_disabled_login_returns_400(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)

    app = FastAPI()
    app.include_router(auth_api.router, prefix="/api/v1")
    client = TestClient(app)

    resp = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 400


def test_auth_enabled_login_and_me(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_USERNAME", "admin")
    monkeypatch.setattr(settings, "AUTH_PASSWORD", "admin123")
    monkeypatch.setattr(settings, "AUTH_TOKEN_TTL_SECONDS", 3600)
    auth_core._sessions.clear()

    app = FastAPI()
    app.include_router(auth_api.router, prefix="/api/v1")
    client = TestClient(app)

    login = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert login.status_code == 200
    token = login.json()["access_token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "admin"


def test_metadata_preview_helper_infers_fields_from_ocr_text():
    text = "Fuji Electric SC-N2S magnetic contactor model"

    assert HybridSearchService._infer_maker_from_text(text) == "Fuji Electric"
    assert HybridSearchService._infer_part_number_from_text(text) == "SC-N2S"
    assert HybridSearchService._infer_category_from_text(text) == "contactor"


def test_confirm_metadata_normalizes_ocr_image_indices():
    indices = HybridSearchService._normalize_ocr_image_indices(["0", 2, -1, "x", 2, 99], 3)
    assert indices == [0, 2]


def test_bulk_item_parser_reads_meta_and_images(tmp_path):
    item_dir = tmp_path / "items" / "auto_part" / "auto_gparts_cat40_000001"
    images_dir = item_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "001.jpg").write_bytes(b"img-1")
    (images_dir / "002.jpg").write_bytes(b"img-2")
    (item_dir / "meta.json").write_text(
        json.dumps(
            {
                "item_id": "auto_gparts_cat40_000001",
                "maker": "Kia",
                "part_number": "98110-2K000",
                "subcategory": "interior_electrical_trim",
                "title": "Wiper motor",
                "description": "Kia wiper motor assembly",
                "product_info": "connector included",
                "price_value": 33000,
            }
        ),
        encoding="utf-8",
    )

    service = HybridSearchService()
    item_dirs = service._discover_bulk_item_dirs(tmp_path)

    assert item_dirs == [item_dir]
    meta = service._load_bulk_item_meta(item_dir)
    metadata = service._normalize_bulk_item_metadata(item_dir, meta)
    images = service._resolve_bulk_item_images(item_dir)

    assert metadata["model_id"] == "auto_gparts_cat40_000001"
    assert metadata["maker"] == "Kia"
    assert metadata["part_number"] == "98110-2K000"
    assert metadata["category"] == "interior_electrical_trim"
    assert "Wiper motor" in metadata["description"]
    assert metadata["price_text"] == "33000"
    assert [path.name for path in images] == ["001.jpg", "002.jpg"]


def test_bulk_zip_extract_blocks_path_traversal(tmp_path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../evil.txt", "nope")

    service = HybridSearchService()

    try:
        service._extract_bulk_zip_archive(archive_path, tmp_path / "out")
    except ValueError as exc:
        assert "unsafe path" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for unsafe zip path")


def test_qwen_metadata_preview_parses_json_response():
    service = HybridSearchService()
    service._qwen_metadata_preview_runtime = SimpleNamespace(
        captioner=SimpleNamespace(
            generate=lambda *_args, **_kwargs: (
                '{"maker":"Fuji Electric","part_number":"SC-N2S","category":"contactor",'
                '"description":"Fuji Electric contactor","product_info":"contactor","price_value":null}'
            )
        )
    )

    draft = service._suggest_metadata_from_qwen([__file__])
    assert draft["source"] == "qwen3_vl"
    assert draft["maker"] == "Fuji Electric"
    assert draft["part_number"] == "SC-N2S"
