from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...core.auth import require_user
from ...schemas.agent import AgentChatRequest, AgentChatResponse
from ...services.agent import agent_service

router = APIRouter(prefix="/agent", tags=["Agent"])

def _extract_sources(debug: dict) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    steps = debug.get("intermediate_steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("tool") != "web_search":
            continue
        obs = step.get("observation")
        if isinstance(obs, list):
            for item in obs:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                if title and url:
                    sources.append({"title": title, "url": url})
    # de-dupe
    uniq: list[dict[str, str]] = []
    seen = set()
    for s in sources:
        key = (s["title"], s["url"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq[:5]


def _extract_identified(debug: dict) -> dict:
    steps = debug.get("intermediate_steps") or []
    best: dict = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("tool") != "hybrid_search":
            continue
        obs = step.get("observation")
        if isinstance(obs, dict):
            if not obs.get("good_match"):
                return {}
            results = obs.get("results")
            if isinstance(results, list) and results and isinstance(results[0], dict) and "model_id" in results[0]:
                return results[0]
    return best


@router.post("/chat", response_model=AgentChatResponse, summary="Agent chat (image + question)")
async def chat(
    request: AgentChatRequest,
    _username: str = Depends(require_user),
) -> AgentChatResponse:
    try:
        request_id = ""
        if request.image_base64:
            request_id = agent_service.put_image(request.image_base64)
        result = await agent_service.chat(
            message=request.message,
            request_id=request_id,
            max_tool_results=request.max_tool_results,
            update_milvus=request.update_milvus,
        )
        answer = str(result.get("output") or "").strip()
        debug = {k: v for k, v in result.items() if k != "output"}
        sources = _extract_sources(debug)
        identified = _extract_identified(debug)
        return AgentChatResponse(
            answer=answer or "답변을 생성하지 못했습니다.",
            sources=sources,
            identified=identified,
            debug=debug,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
