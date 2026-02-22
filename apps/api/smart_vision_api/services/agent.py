from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from ..core.config import settings
from .catalog import catalog_service
from .gparts import search_prices
from .hybrid import hybrid_service
from .web_search import duckduckgo_search, extract_price_mentions


@dataclass(frozen=True)
class _StoredImage:
    image_base64: str
    expires_at: float


class _ImageStore:
    def __init__(self, *, ttl_seconds: int = 600, max_items: int = 128) -> None:
        self._ttl = ttl_seconds
        self._max = max_items
        self._items: Dict[str, _StoredImage] = {}

    def put(self, image_base64: str) -> str:
        now = time.time()
        self._gc(now)
        if len(self._items) >= self._max:
            # Remove oldest by expiry (simple)
            oldest = sorted(self._items.items(), key=lambda kv: kv[1].expires_at)[0][0]
            self._items.pop(oldest, None)
        request_id = uuid.uuid4().hex
        self._items[request_id] = _StoredImage(image_base64=image_base64, expires_at=now + self._ttl)
        return request_id

    def get(self, request_id: str) -> Optional[str]:
        now = time.time()
        self._gc(now)
        item = self._items.get(request_id)
        if item is None:
            return None
        if now >= item.expires_at:
            self._items.pop(request_id, None)
            return None
        return item.image_base64

    def _gc(self, now: float) -> None:
        expired = [k for k, v in self._items.items() if now >= v.expires_at]
        for k in expired:
            self._items.pop(k, None)


class SmartVisionAgentService:
    """Tool-calling agent for retrieval + open-world enrichment.

    Uses langchain_openai tool-calling with a lightweight control loop (no
    dependency on langchain.agents public API, which is version-volatile).
    """

    def __init__(self) -> None:
        self._image_store = _ImageStore()

    def put_image(self, image_base64: str) -> str:
        return self._image_store.put(image_base64)

    def _require_openai_key(self) -> None:
        import os

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Configure it to use the agent.")

    def _build_tools(self, *, allow_updates: bool):
        match_threshold = 0.75

        @tool("hybrid_search")
        def hybrid_search(request_id: str = "", query_text: str = "", top_k: int = 5) -> dict[str, Any]:
            """Run Smart Vision hybrid search. Use request_id if the user provided an image.

            The agent should treat matches as "good" only when top score >= 0.75.
            """
            image_b64 = None
            rid = (request_id or "").strip()
            if rid:
                image_b64 = self._image_store.get(rid)
                if image_b64 is None:
                    return {"error": "image request_id not found or expired", "good_match": False, "results": []}
            q = (query_text or "").strip() or None
            top_k = int(top_k or 5)
            top_k = max(1, min(20, top_k))
            results = hybrid_service.search(query_text=q, image_b64=image_b64, top_k=top_k, part_number=None)
            top_score = None
            if isinstance(results, list) and results and isinstance(results[0], dict):
                try:
                    top_score = float(results[0].get("score")) if results[0].get("score") is not None else None
                except (TypeError, ValueError):
                    top_score = None
            good_match = bool(top_score is not None and top_score >= match_threshold)
            return {
                "threshold": match_threshold,
                "good_match": good_match,
                "top_score": top_score,
                "results": results,
            }

        @tool("vision_identify")
        def vision_identify(request_id: str) -> dict[str, Any]:
            """Use a vision-capable LLM to describe the uploaded image and guess product keywords."""
            rid = (request_id or "").strip()
            if not rid:
                return {"error": "request_id required"}
            image_b64 = self._image_store.get(rid)
            if image_b64 is None:
                return {"error": "image request_id not found or expired"}
            vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=60)
            msg = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "이 이미지를 보고 어떤 부품/제품인지 최대한 구체적으로 추정해줘.\n"
                            "가능하면 maker/모델명/부품명/키워드/추정 근거를 간단히 적어줘.\n"
                            "확신이 낮으면 불확실하다고 말해."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]
            )
            out = vision_llm.invoke([msg]).content
            return {"description": str(out)}

        @tool("web_search")
        def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
            """Search the public web for information about the identified product."""
            max_results = int(max_results or 5)
            max_results = max(1, min(10, max_results))
            results = duckduckgo_search(query=query, max_results=max_results)
            return [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]

        @tool("extract_prices")
        def extract_prices(text: str) -> list[str]:
            """Extract price mentions from a text blob (snippets/pages)."""
            return extract_price_mentions(text)

        @tool("allocate_model_id")
        def allocate_model_id(category: str = "", prefix: str = "", width: int = 6) -> dict[str, Any]:
            """Allocate a new sequential model_id (e.g., a000001) by category prefix."""
            category = (category or "").strip()
            prefix = (prefix or "").strip()
            model_id = hybrid_service.orchestrator.allocate_model_id(category=category or None, prefix=prefix or None, width=int(width or 6))
            return {"model_id": model_id}

        @tool("upsert_model_metadata")
        def upsert_model_metadata(
            model_id: str,
            maker: str = "",
            part_number: str = "",
            category: str = "",
            description: str = "",
            web_text: str = "",
            price_text: str = "",
        ) -> dict[str, Any]:
            """Create/update a model record in Milvus with enriched metadata fields."""
            if not allow_updates:
                return {"error": "milvus updates disabled for this request"}
            model_id = (model_id or "").strip()
            if not model_id:
                return {"error": "model_id required"}
            payload: Dict[str, str] = {"model_id": model_id}
            if maker.strip():
                payload["maker"] = maker.strip()
            if part_number.strip():
                payload["part_number"] = part_number.strip()
            if category.strip():
                payload["category"] = category.strip()
            if description.strip():
                payload["description"] = description.strip()
            if web_text.strip():
                payload["web_text"] = web_text.strip()[:6000]
            if price_text.strip():
                payload["price_text"] = price_text.strip()[:1000]
            row = hybrid_service.orchestrator.index_model_metadata(model_id, payload)
            return {"status": "upserted", "model_id": model_id, "stored": bool(row)}

        @tool("gparts_search_prices")
        def gparts_search_prices(keyword: str, max_results: int = 5) -> list[dict[str, Any]]:
            """Search gparts.co.kr by keyword and return price candidates (KRW)."""
            max_results = int(max_results or 5)
            max_results = max(1, min(10, max_results))
            items = search_prices(keyword=keyword, max_results=max_results)
            return [
                {
                    "goods_cd": it.goods_cd,
                    "title": it.title,
                    "price_krw": it.price_krw,
                    "detail_url": it.detail_url,
                    "thumbnail_url": it.thumbnail_url,
                }
                for it in items
            ]

        @tool("catalog_search")
        def catalog_search(query: str, top_k: int = 5, model_id: str = "", part_number: str = "") -> dict[str, Any]:
            """Search internal catalog RAG chunks indexed from company PDF documents."""
            top_k = int(top_k or 5)
            top_k = max(1, min(20, top_k))
            results = catalog_service.search(
                query_text=query,
                top_k=top_k,
                model_id=(model_id or "").strip() or None,
                part_number=(part_number or "").strip() or None,
            )
            return {"results": results, "count": len(results)}

        tools = [
            hybrid_search,
            vision_identify,
            web_search,
            extract_prices,
            allocate_model_id,
            upsert_model_metadata,
            gparts_search_prices,
            catalog_search,
        ]
        return tools

    async def chat(
        self,
        *,
        message: str,
        request_id: str,
        max_tool_results: int,
        update_milvus: bool,
    ) -> Dict[str, Any]:
        self._require_openai_key()

        allow_updates = bool(update_milvus)
        tools = self._build_tools(allow_updates=allow_updates)
        tool_map = {t.name: t for t in tools}

        system_prompt = (
            "You are Smart Vision, a Korean assistant for identifying parts from an uploaded image and answering questions.\n"
            "You have tools:\n"
            "- hybrid_search(request_id, query_text, top_k)\n"
            "- vision_identify(request_id)\n"
            "- web_search(query, max_results)\n"
            "- extract_prices(text)\n"
            "- allocate_model_id(category, prefix, width)\n"
            "- upsert_model_metadata(model_id, maker, part_number, category, description, web_text, price_text)\n"
            "- gparts_search_prices(keyword, max_results)  (example source)\n\n"
            "- catalog_search(query, top_k, model_id, part_number)\n\n"
            f"request_id={request_id}\n"
            f"update_milvus={'true' if allow_updates else 'false'}\n\n"
            "Rules:\n"
            "1) If an image is provided, call hybrid_search first. Treat it as a match only if good_match=true (threshold=0.75). If no good match, call vision_identify.\n"
            "2) For internal product specs/manual questions, call catalog_search first.\n"
            "3) For open-world info/price, use web_search and extract_prices.\n"
            "4) If there's no model_id (no match) and update_milvus=true, allocate_model_id by category prefix and upsert_model_metadata.\n"
            "5) If you used web_search, include 2-5 sources (title + url) at the end.\n"
            "6) If you used catalog_search, cite source and page in the answer.\n"
            "7) If request_id is non-empty, never ask the user to upload an image again in this turn.\n"
            "Be honest about uncertainty.\n"
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=60).bind_tools(tools)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]

        intermediate_steps = []
        max_iters = 6
        for _ in range(max_iters):
            ai = llm.invoke(messages)
            messages.append(ai)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                # Fallback: when an image is present but the model skipped tool-calls,
                # inject a deterministic hybrid search context once and retry.
                if request_id and not intermediate_steps:
                    forced_tool = tool_map.get("hybrid_search")
                    if forced_tool is not None:
                        forced_args = {
                            "request_id": request_id,
                            "query_text": message,
                            "top_k": max_tool_results,
                        }
                        try:
                            forced_obs = forced_tool.invoke(forced_args)
                        except Exception as exc:
                            forced_obs = {"error": str(exc)}
                        intermediate_steps.append(
                            {"tool": "hybrid_search", "args": forced_args, "observation": forced_obs}
                        )
                        messages.append(
                            HumanMessage(
                                content=(
                                    "The user already uploaded an image. "
                                    "Use this hybrid_search result and answer directly.\n"
                                    f"{json.dumps(forced_obs, ensure_ascii=False)}"
                                )
                            )
                        )
                        continue
                break
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args") or {}
                tool_call_id = call.get("id")
                tool_obj = tool_map.get(name)
                if tool_obj is None:
                    obs = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        obs = tool_obj.invoke(args)
                    except Exception as exc:
                        obs = {"error": str(exc)}
                intermediate_steps.append({"tool": name, "args": args, "observation": obs})
                messages.append(ToolMessage(content=json.dumps(obs, ensure_ascii=False), tool_call_id=tool_call_id))

        output_text = ""
        # The last message should be an AIMessage with final content.
        for m in reversed(messages):
            if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip():
                output_text = m.content.strip()
                break

        return {"output": output_text, "intermediate_steps": intermediate_steps}


agent_service = SmartVisionAgentService()
