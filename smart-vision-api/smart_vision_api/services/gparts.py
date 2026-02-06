from __future__ import annotations

import dataclasses
import re
import time
import urllib.parse
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


BASE_URL = "http://www.gparts.co.kr"
SEARCH_PATH = "/display/showPartList.do"
DETAIL_PATH = "/goods/viewGoodsInfo.do"


@dataclasses.dataclass(frozen=True)
class GpartsSearchItem:
    goods_cd: str
    title: str
    price_krw: Optional[int]
    detail_url: str
    thumbnail_url: Optional[str]


_PRICE_RE = re.compile(r"(\d{1,3}(?:,\d{3})+)\s*원")
_GOODS_RE = re.compile(r"goodsCd\s*:\s*'(\d+)'")


def _desktop_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Referer": urllib.parse.urljoin(BASE_URL, "/main.do"),
    }


def _fix_mobile_redirect(url: str) -> str:
    # Some servers redirect unknown clients to m.gparts.co.kr with double slashes.
    return url.replace("http://m.gparts.co.kr//", "http://www.gparts.co.kr/")


def search_prices(
    *,
    keyword: str,
    page_idx: int = 1,
    page_size: int = 30,
    max_results: int = 5,
    timeout_s: float = 30.0,
    sleep_s: float = 0.25,
) -> List[GpartsSearchItem]:
    """Search gparts list page and extract price/title for top items.

    This scrapes server-rendered HTML; it may break if the site changes.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    time.sleep(max(0.0, sleep_s))

    url = urllib.parse.urljoin(BASE_URL, SEARCH_PATH)
    data = {"searchKeyword": keyword, "pageIdx": str(page_idx), "pageCnt": str(page_size)}
    session = requests.Session()

    resp = session.post(url, data=data, headers=_desktop_headers(), timeout=timeout_s, allow_redirects=False)
    if resp.status_code in {301, 302, 303, 307, 308}:
        location = (resp.headers.get("location") or "").strip()
        resp = session.post(_fix_mobile_redirect(location), data=data, headers=_desktop_headers(), timeout=timeout_s)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    container = soup.select_one("div.resultList#listType") or soup.select_one("div.resultList")
    if container is None:
        return []

    results: List[GpartsSearchItem] = []
    for li in container.select("ul > li"):
        if len(results) >= max_results:
            break

        a = li.select_one("a[onclick]")
        onclick = (a.get("onclick") if a else "") or ""
        m_goods = _GOODS_RE.search(onclick)
        if not m_goods:
            continue
        goods_cd = m_goods.group(1)

        title_el = li.select_one(".detail .tit") or li.select_one(".tit")
        title = title_el.get_text(" ", strip=True) if title_el else li.get_text(" ", strip=True)
        title = (title or "").strip()

        thumbnail = None
        img = li.select_one("img.thumbnail")
        if img and img.get("src"):
            thumbnail = img.get("src").strip()

        # Price is usually present as "xxx원" inside the list item text.
        price_krw = None
        m_price = _PRICE_RE.search(li.get_text(" ", strip=True))
        if m_price:
            price_krw = int(m_price.group(1).replace(",", ""))

        detail_url = urllib.parse.urljoin(BASE_URL, DETAIL_PATH) + "?" + urllib.parse.urlencode({"goodsCd": goods_cd})
        results.append(
            GpartsSearchItem(
                goods_cd=goods_cd,
                title=title,
                price_krw=price_krw,
                detail_url=detail_url,
                thumbnail_url=thumbnail,
            )
        )

    return results

