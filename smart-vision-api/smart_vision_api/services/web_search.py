from __future__ import annotations

import dataclasses
import html
import re
import urllib.parse
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


@dataclasses.dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str


def duckduckgo_search(
    *,
    query: str,
    max_results: int = 5,
    timeout_s: float = 20.0,
) -> List[WebSearchResult]:
    """Lightweight HTML scraping search.

    Uses DuckDuckGo's HTML endpoint. This can break if markup changes.
    """
    query = (query or "").strip()
    if not query:
        return []
    max_results = max(1, min(int(max_results or 5), 10))

    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    resp = requests.post(url, data=params, headers=headers, timeout=timeout_s)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[WebSearchResult] = []
    for item in soup.select("div.result"):
        if len(results) >= max_results:
            break
        a = item.select_one("a.result__a")
        if not a or not a.get("href"):
            continue
        title = a.get_text(" ", strip=True)
        href = a.get("href")
        snippet_el = item.select_one(".result__snippet") or item.select_one("a.result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""

        href = href.strip()
        # Sometimes DDG wraps redirect URLs; try to unwrap.
        parsed = urllib.parse.urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
            qs = urllib.parse.parse_qs(parsed.query)
            target = qs.get("uddg", [None])[0]
            if target:
                href = urllib.parse.unquote(target)

        title = html.unescape(title)
        snippet = html.unescape(snippet)
        results.append(WebSearchResult(title=title, url=href, snippet=snippet))

    return results


_KRW_RE = re.compile(r"(\d{1,3}(?:,\d{3})+)\s*원")
_USD_RE = re.compile(r"\\$\\s*(\\d{1,3}(?:,\\d{3})+|\\d+(?:\\.\\d+)?)")


def extract_price_mentions(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    mentions: List[str] = []
    for m in _KRW_RE.findall(text):
        mentions.append(f"{m}원")
    for m in _USD_RE.findall(text):
        mentions.append(f"${m}")
    # de-dupe preserving order
    out = []
    seen = set()
    for x in mentions:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out[:10]

