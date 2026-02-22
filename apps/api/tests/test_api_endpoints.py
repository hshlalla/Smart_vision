from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from smart_vision_api.api.v1 import auth as auth_api
from smart_vision_api.api.v1 import catalog as catalog_api
from smart_vision_api.api.v1 import hybrid as hybrid_api
from smart_vision_api.core import auth as auth_core
from smart_vision_api.core.auth import require_user
from smart_vision_api.core.config import settings


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
