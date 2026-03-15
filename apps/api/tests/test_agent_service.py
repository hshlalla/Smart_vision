from __future__ import annotations

import pytest

from smart_vision_api.services.agent import SmartVisionAgentService


@pytest.mark.anyio
async def test_agent_service_uses_fast_path_for_text_lookup(monkeypatch):
    service = SmartVisionAgentService()

    def fake_search(**kwargs):
        assert kwargs["query_text"] == "Fuji electric fa 찾아줘"
        assert kwargs["image_b64"] is None
        return [
            {
                "model_id": "f000123",
                "score": 0.91,
                "maker": "Fuji Electric",
                "part_number": "FA-100",
                "category": "contactor",
                "description": "Fuji Electric FA series contactor",
                "lexical_hit": True,
                "exact_field_boost": 0.30,
            }
        ]

    monkeypatch.setattr("smart_vision_api.services.agent.hybrid_service.search", fake_search)

    result = await service.chat(
        message="Fuji electric fa 찾아줘",
        request_id="",
        max_tool_results=5,
        update_milvus=False,
    )

    assert "Fuji Electric" in result["output"]
    assert result["intermediate_steps"][0]["tool"] == "hybrid_search"
    assert result["intermediate_steps"][0]["observation"]["fast_path"] is True


@pytest.mark.anyio
async def test_agent_service_normalizes_part_number_for_fast_path(monkeypatch):
    service = SmartVisionAgentService()

    def fake_search(**kwargs):
        assert kwargs["query_text"] == "91200 4F310"
        assert kwargs["part_number"] == "912004F310"
        return [
            {
                "model_id": "h000001",
                "score": 0.61,
                "maker": "Hyundai",
                "part_number": "91200-4F310",
                "category": "motor",
                "description": "Hyundai motor assembly",
                "lexical_hit": False,
                "exact_field_boost": 0.0,
            }
        ]

    monkeypatch.setattr("smart_vision_api.services.agent.hybrid_service.search", fake_search)

    result = await service.chat(
        message="91200 4F310",
        request_id="",
        max_tool_results=5,
        update_milvus=False,
    )

    assert "91200-4F310" in result["output"]
    assert result["intermediate_steps"][0]["observation"]["part_number_candidate"] == "912004F310"


@pytest.mark.anyio
async def test_agent_service_returns_collection_stats(monkeypatch):
    service = SmartVisionAgentService()

    def fake_collection_stats():
        return {
            "image": {"name": "qwen3_vl_image_parts", "num_entities": 3, "exists": True},
            "text": {"name": "bge_m3_text_parts", "num_entities": 3, "exists": True},
            "attrs": {"name": "attrs_parts_v2", "num_entities": 3, "exists": True},
            "model": {"name": "bge_m3_model_texts", "num_entities": 1, "exists": True},
            "caption": {"name": "bge_m3_caption_parts", "num_entities": 3, "exists": True},
        }

    monkeypatch.setattr("smart_vision_api.services.agent.hybrid_service.collection_stats", fake_collection_stats)

    result = await service.chat(
        message="지금 저장되어 있는 콜렉션에 내용 몇개인지 알려줘",
        request_id="",
        max_tool_results=5,
        update_milvus=False,
    )

    assert "현재 Milvus 컬렉션 엔티티 수는 다음과 같습니다." in result["output"]
    assert "qwen3_vl_image_parts: 3" in result["output"]
    assert result["intermediate_steps"][0]["tool"] == "collection_stats"
