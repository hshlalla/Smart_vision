from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from ..core.config import settings
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
    """LangChain tool-calling agent that can do retrieval + price lookup."""

    def __init__(self) -> None:
        self._image_store = _ImageStore()
        self._executor: Optional[AgentExecutor] = None

    def put_image(self, image_base64: str) -> str:
        return self._image_store.put(image_base64)

    def _build_executor(self) -> AgentExecutor:
        import os

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Configure it to use the agent.")

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=60)

        @tool("hybrid_search")
        def hybrid_search(request_id: str = "", query_text: str = "", top_k: int = 5) -> list[dict[str, Any]]:
            """Run Smart Vision hybrid search. Use request_id if the user provided an image."""
            image_b64 = None
            rid = (request_id or "").strip()
            if rid:
                image_b64 = self._image_store.get(rid)
                if image_b64 is None:
                    return [{"error": "image request_id not found or expired"}]
            q = (query_text or "").strip() or None
            top_k = int(top_k or 5)
            top_k = max(1, min(20, top_k))
            return hybrid_service.search(query_text=q, image_b64=image_b64, top_k=top_k, part_number=None)

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

        @tool("update_milvus_enrichment")
        def update_milvus_enrichment(model_id: str, web_text: str = "", price_text: str = "") -> dict[str, Any]:
            """Persist enriched info back into Milvus (updates model metadata_text vector)."""
            model_id = (model_id or "").strip()
            if not model_id:
                return {"error": "model_id required"}
            payload: Dict[str, str] = {"model_id": model_id}
            if web_text and web_text.strip():
                payload["web_text"] = web_text.strip()[:6000]
            if price_text and price_text.strip():
                payload["price_text"] = price_text.strip()[:1000]
            row = hybrid_service.orchestrator.index_model_metadata(model_id, payload)
            return {"status": "updated", "model_id": model_id, "stored": bool(row)}

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

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are Smart Vision, a Korean assistant for identifying parts from an uploaded image and answering questions.\n"
                    "Use tools when needed:\n"
                    "- If the user provided an image, first call hybrid_search with request_id.\n"
                    "- If hybrid_search returns no good candidates, call vision_identify.\n"
                    "- For open-world info/price, use web_search then extract_prices on snippets.\n"
                    "- Only if update_milvus=true AND you used web info AND you have a model_id from hybrid_search, call update_milvus_enrichment to persist.\n"
                    "- gparts_search_prices is only an example source; prefer web_search unless the user explicitly asks for gparts.\n"
                    "Be honest about uncertainty. Ask a short follow-up question when ambiguous.\n"
                    "When you use web_search results, include 2-5 sources (title + url) at the end.\n"
                    "Return concise, practical answers in Korean.\n",
                ),
                ("human", "request_id={request_id}\nupdate_milvus={update_milvus}\nmax_tool_results={max_tool_results}\n\nUser: {input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        tools = [hybrid_search, vision_identify, web_search, extract_prices, update_milvus_enrichment, gparts_search_prices]
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=6,
            return_intermediate_steps=True,
        )

    @property
    def executor(self) -> AgentExecutor:
        if self._executor is None:
            self._executor = self._build_executor()
        return self._executor

    async def chat(
        self,
        *,
        message: str,
        request_id: str,
        max_tool_results: int,
        update_milvus: bool,
    ) -> Dict[str, Any]:
        # max_tool_results currently enforced inside tools; keep for future shaping
        result = await self.executor.ainvoke(
            {
                "input": message,
                "request_id": request_id,
                "update_milvus": bool(update_milvus),
                "max_tool_results": int(max_tool_results),
            }
        )
        return result


agent_service = SmartVisionAgentService()
