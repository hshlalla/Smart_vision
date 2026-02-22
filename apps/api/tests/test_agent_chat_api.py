from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from smart_vision_api.api.v1 import agent as agent_api
from smart_vision_api.core.auth import require_user
from smart_vision_api.core.config import settings


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(agent_api.router, prefix="/api/v1")
    app.dependency_overrides[require_user] = lambda: "tester"
    return TestClient(app)


def test_agent_chat_success_and_source_dedupe(monkeypatch):
    async def fake_chat(**kwargs):
        return {
            "output": "테스트 답변",
            "intermediate_steps": [
                {
                    "tool": "web_search",
                    "observation": [
                        {"title": "A", "url": "https://a.example", "snippet": "x"},
                        {"title": "A", "url": "https://a.example", "snippet": "dup"},
                        {"title": "B", "url": "https://b.example", "snippet": "y"},
                    ],
                }
            ],
        }

    monkeypatch.setattr(agent_api.agent_service, "chat", fake_chat)
    client = _build_client()

    resp = client.post("/api/v1/agent/chat", json={"message": "안녕"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "테스트 답변"
    assert body["sources"] == [
        {"title": "A", "url": "https://a.example"},
        {"title": "B", "url": "https://b.example"},
    ]


def test_agent_chat_adds_catalog_evidence_when_missing(monkeypatch):
    async def fake_chat(**kwargs):
        return {
            "output": "카탈로그 기반 답변",
            "intermediate_steps": [
                {
                    "tool": "catalog_search",
                    "observation": {
                        "results": [
                            {"source": "manual.pdf", "page": 12},
                            {"source": "manual.pdf", "page": 12},
                        ]
                    },
                }
            ],
        }

    monkeypatch.setattr(agent_api.agent_service, "chat", fake_chat)
    client = _build_client()

    resp = client.post("/api/v1/agent/chat", json={"message": "사양 알려줘"})
    assert resp.status_code == 200
    answer = resp.json()["answer"]
    assert "Catalog Evidence:" in answer
    assert "- manual.pdf (page 12)" in answer


def test_agent_chat_passes_request_id_for_image(monkeypatch):
    seen = {"request_id": None}

    def fake_put_image(image_b64: str) -> str:
        assert image_b64 == "ZmFrZQ=="
        return "rid-123"

    async def fake_chat(**kwargs):
        seen["request_id"] = kwargs["request_id"]
        return {"output": "ok", "intermediate_steps": []}

    monkeypatch.setattr(agent_api.agent_service, "put_image", fake_put_image)
    monkeypatch.setattr(agent_api.agent_service, "chat", fake_chat)
    client = _build_client()

    resp = client.post(
        "/api/v1/agent/chat",
        json={"message": "이미지 확인", "image_base64": "ZmFrZQ=="},
    )
    assert resp.status_code == 200
    assert seen["request_id"] == "rid-123"


def test_agent_chat_returns_500_detail_on_service_error(monkeypatch):
    async def fake_chat(**kwargs):
        raise RuntimeError("OPENAI_API_KEY is not set. Configure it to use the agent.")

    monkeypatch.setattr(agent_api.agent_service, "chat", fake_chat)
    client = _build_client()

    resp = client.post("/api/v1/agent/chat", json={"message": "안녕"})
    assert resp.status_code == 500
    assert "OPENAI_API_KEY is not set" in resp.json()["detail"]


def test_agent_chat_rejects_too_large_image(monkeypatch):
    monkeypatch.setattr(settings, "MAX_IMAGE_BASE64_LENGTH", 10)
    client = _build_client()

    resp = client.post(
        "/api/v1/agent/chat",
        json={"message": "이미지", "image_base64": "a" * 11},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"]
