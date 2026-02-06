#!/usr/bin/env python3
"""
Gparts image crawler (keyword-based).

This script crawls product image URLs from gparts.co.kr search results and,
optionally, downloads the images.

Notes
-----
- Be mindful of the website's Terms/robots.txt and crawl politely (rate limit).
- The site search is server-rendered; we scrape list pages and then each goods
  detail page to collect all image URLs (usually hosted on S3).

Example
-------
List + write metadata only (default):
python3 dataset/crawler.py --keyword "헤드램프" --max-pages 3 --out dataset/gparts_headlamp

Download images too:
python3 dataset/crawler.py --keyword "헤드램프" --max-pages 3 --download --out dataset/gparts_headlamp

Resume (skip already-seen goods):
python3 dataset/crawler.py --keyword "헤드램프" --max-pages 3 --resume --out dataset/gparts_headlamp
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from pathlib import Path
from typing import Iterable, Iterator, Optional

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: requests. Install with `pip install requests`.") from exc

try:
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: beautifulsoup4. Install with `pip install beautifulsoup4`.") from exc


BASE_URL = "http://www.gparts.co.kr"
SEARCH_PATH = "/display/showPartList.do"
DETAIL_PATH = "/goods/viewGoodsInfo.do"
DEFAULT_PAGE_SIZE = 30


@dataclasses.dataclass(frozen=True)
class GoodsListItem:
    goods_cd: str
    title: str
    thumbnail_url: str | None
    page_idx: int


@dataclasses.dataclass(frozen=True)
class GoodsDetail:
    goods_cd: str
    detail_url: str
    image_urls: list[str]


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _build_session(*, user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Connection": "keep-alive",
        }
    )
    return session


def _load_robots(session: requests.Session, *, base_url: str, user_agent: str) -> urllib.robotparser.RobotFileParser:
    robots_url = urllib.parse.urljoin(base_url, "/robots.txt")
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        resp = session.get(robots_url, timeout=30)
        if resp.status_code == 200:
            parser.parse(resp.text.splitlines())
        else:  # treat as no rules
            parser.parse([])
    except Exception:
        parser.parse([])
    # robotparser doesn't store UA; caller passes UA to can_fetch via second arg
    return parser


def _sleep_polite(base_seconds: float, *, jitter: float) -> None:
    if base_seconds <= 0:
        return
    delay = base_seconds + random.random() * max(0.0, jitter)
    time.sleep(delay)


def _extract_max_page_idx(html: str) -> int:
    pages = [int(x) for x in re.findall(r"fnPaginate\((\d+)\)", html)]
    return max(pages) if pages else 1


def fetch_search_page(
    session: requests.Session,
    *,
    keyword: str,
    page_idx: int,
    page_size: int,
    timeout_s: float,
    base_url: str = BASE_URL,
) -> str:
    url = urllib.parse.urljoin(base_url, SEARCH_PATH)
    data = {
        "searchKeyword": keyword,
        "pageIdx": str(page_idx),
        "pageCnt": str(page_size),
    }
    resp = session.post(
        url,
        data=data,
        timeout=timeout_s,
        headers={"Referer": urllib.parse.urljoin(base_url, "/main.do")},
        allow_redirects=False,
    )
    if resp.status_code in {301, 302, 303, 307, 308}:
        location = (resp.headers.get("location") or "").strip()
        # gparts sometimes redirects unknown UAs to the mobile domain with a double-slash path.
        # Example: http://m.gparts.co.kr//display/showPartList.do
        if "m.gparts.co.kr" in location and "//display/" in location:
            fixed = location.replace("http://m.gparts.co.kr//", "http://www.gparts.co.kr/")
            resp = session.post(
                fixed,
                data=data,
                timeout=timeout_s,
                headers={"Referer": urllib.parse.urljoin(base_url, "/main.do")},
            )
        else:
            resp = session.post(
                urllib.parse.urljoin(base_url, location),
                data=data,
                timeout=timeout_s,
                headers={"Referer": urllib.parse.urljoin(base_url, "/main.do")},
            )
    resp.raise_for_status()
    return resp.text


def parse_goods_list(html: str, *, page_idx: int) -> list[GoodsListItem]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[GoodsListItem] = []

    # The list is usually in: <div class="resultList" id="listType"><ul><li>...</li></ul></div>
    container = soup.select_one("div.resultList#listType") or soup.select_one("div.resultList")
    if container is None:
        return []

    for li in container.select("ul > li"):
        a = li.select_one("a[onclick]")
        onclick = (a.get("onclick") if a else "") or ""
        m = re.search(r"goodsCd\s*:\s*'(\d+)'", onclick)
        if not m:
            continue
        goods_cd = m.group(1)
        title = ""
        title_el = li.select_one(".detail .tit") or li.select_one(".tit")
        if title_el:
            title = title_el.get_text(" ", strip=True)

        thumb = None
        img = li.select_one("img.thumbnail")
        if img and img.get("src"):
            thumb = img.get("src").strip()

        items.append(GoodsListItem(goods_cd=goods_cd, title=title, thumbnail_url=thumb, page_idx=page_idx))

    return items


def fetch_goods_detail_html(
    session: requests.Session,
    *,
    goods_cd: str,
    timeout_s: float,
    base_url: str = BASE_URL,
) -> str:
    url = urllib.parse.urljoin(base_url, DETAIL_PATH)
    resp = session.get(
        url,
        params={"goodsCd": goods_cd},
        timeout=timeout_s,
        headers={"Referer": urllib.parse.urljoin(base_url, SEARCH_PATH)},
        allow_redirects=False,
    )
    if resp.status_code in {301, 302, 303, 307, 308}:
        location = (resp.headers.get("location") or "").strip()
        if "m.gparts.co.kr" in location and "//goods/" in location:
            fixed = location.replace("http://m.gparts.co.kr//", "http://www.gparts.co.kr/")
            resp = session.get(
                fixed,
                params={"goodsCd": goods_cd},
                timeout=timeout_s,
                headers={"Referer": urllib.parse.urljoin(base_url, SEARCH_PATH)},
            )
        else:
            resp = session.get(
                urllib.parse.urljoin(base_url, location),
                params={"goodsCd": goods_cd},
                timeout=timeout_s,
                headers={"Referer": urllib.parse.urljoin(base_url, SEARCH_PATH)},
            )
    resp.raise_for_status()
    return resp.text


def parse_goods_detail_images(html: str, *, include_thumbnails: bool) -> list[str]:
    # Product images are usually absolute S3 URLs embedded in img tags.
    urls = set(re.findall(r"https?://[^\s\"']+\.(?:jpg|jpeg|png|webp)", html, flags=re.IGNORECASE))
    # Keep only obvious product image hosts/paths; drop site meta images.
    filtered = [u for u in urls if "amazonaws.com/gparts/productImage/" in u]
    if not include_thumbnails:
        filtered = [u for u in filtered if "_400x300" not in u]
    # Stable ordering by the numeric filename if present (1.jpg, 2.jpg, ...)
    def _key(u: str) -> tuple[int, str]:
        name = u.rsplit("/", 1)[-1]
        m = re.match(r"(\d+)\.", name)
        return (int(m.group(1)) if m else 10**9, u)

    return sorted(set(filtered), key=_key)


def _safe_filename(name: str) -> str:
    name = name.strip().replace("\x00", "")
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name or "file"


def download_file(
    session: requests.Session,
    *,
    url: str,
    dest_path: Path,
    timeout_s: float,
    chunk_size: int = 1024 * 256,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    with session.get(url, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, dest_path)


def iter_goods_from_keyword(
    session: requests.Session,
    *,
    keyword: str,
    page_size: int,
    max_pages: Optional[int],
    timeout_s: float,
    sleep_s: float,
    jitter_s: float,
) -> Iterator[tuple[int, int, list[GoodsListItem]]]:
    html = fetch_search_page(session, keyword=keyword, page_idx=1, page_size=page_size, timeout_s=timeout_s)
    last_page = _extract_max_page_idx(html)
    if max_pages is not None:
        last_page = min(last_page, max_pages)
    items = parse_goods_list(html, page_idx=1)
    yield 1, last_page, items

    for page_idx in range(2, last_page + 1):
        _sleep_polite(sleep_s, jitter=jitter_s)
        html = fetch_search_page(session, keyword=keyword, page_idx=page_idx, page_size=page_size, timeout_s=timeout_s)
        items = parse_goods_list(html, page_idx=page_idx)
        yield page_idx, last_page, items


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Crawl and optionally download gparts.co.kr product images by keyword.")
    parser.add_argument("--keyword", required=True, help="Search keyword (e.g., 헤드램프).")
    parser.add_argument("--out", default="dataset/gparts_crawl", help="Output directory for downloads/metadata.")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Items per page (site default is 30).")
    parser.add_argument("--max-pages", type=int, default=1, help="Max number of pages to crawl (safety limit).")
    parser.add_argument("--max-items", type=int, default=0, help="Max number of goods items to process (0 = unlimited).")
    parser.add_argument("--download", action="store_true", help="Actually download images (otherwise only write metadata).")
    parser.add_argument("--include-thumbs", action="store_true", help="Include *_400x300.jpg thumbnail variants.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing jsonl (skip already-seen goods_cd).",
    )
    parser.add_argument("--sleep", type=float, default=0.7, help="Base sleep seconds between requests.")
    parser.add_argument("--jitter", type=float, default=0.4, help="Random jitter seconds added to --sleep.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout seconds.")
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        help="User-Agent header (desktop UA recommended; otherwise site may redirect to m.gparts.co.kr).",
    )
    parser.add_argument("--no-robots", action="store_true", help="Do not check robots.txt (not recommended).")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except downloads and file writes.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).expanduser().resolve()
    keyword = str(args.keyword).strip()
    if not keyword:
        raise SystemExit("--keyword must be non-empty.")

    session = _build_session(user_agent=args.user_agent)

    robots = None
    if not args.no_robots:
        robots = _load_robots(session, base_url=BASE_URL, user_agent=args.user_agent)
        # Check the specific endpoints we will hit.
        for path in (SEARCH_PATH, DETAIL_PATH):
            url = urllib.parse.urljoin(BASE_URL, path)
            if not robots.can_fetch(args.user_agent, url):
                raise SystemExit(f"robots.txt disallows crawling: {url}")

    meta_path = out_dir / "gparts_items.jsonl"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    seen_goods: set[str] = set()
    if args.resume and not args.dry_run and meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    goods_cd = str(obj.get("goods_cd") or "").strip()
                    if goods_cd:
                        seen_goods.add(goods_cd)
        except Exception as exc:
            print(f"[warn] failed to read resume file {meta_path}: {exc}", file=sys.stderr)

    reached_limit = False
    for page_idx, last_page, items in iter_goods_from_keyword(
        session,
        keyword=keyword,
        page_size=args.page_size,
        max_pages=args.max_pages,
        timeout_s=args.timeout,
        sleep_s=args.sleep,
        jitter_s=args.jitter,
    ):
        if not items:
            print(f"[page {page_idx}/{last_page}] no items", file=sys.stderr)
            continue

        print(f"[page {page_idx}/{last_page}] items={len(items)}", file=sys.stderr)

        for item in items:
            if item.goods_cd in seen_goods:
                continue
            seen_goods.add(item.goods_cd)

            processed += 1
            if args.max_items and processed > args.max_items:
                reached_limit = True
                break

            _sleep_polite(args.sleep, jitter=args.jitter)
            detail_html = fetch_goods_detail_html(session, goods_cd=item.goods_cd, timeout_s=args.timeout)
            detail_url = urllib.parse.urljoin(BASE_URL, DETAIL_PATH) + "?" + urllib.parse.urlencode({"goodsCd": item.goods_cd})
            images = parse_goods_detail_images(detail_html, include_thumbnails=args.include_thumbs)
            detail = GoodsDetail(goods_cd=item.goods_cd, detail_url=detail_url, image_urls=images)

            record = {
                "crawled_at": _utc_now_iso(),
                "source": "gparts.co.kr",
                "keyword": keyword,
                "page_idx": item.page_idx,
                "goods_cd": item.goods_cd,
                "title": item.title,
                "thumbnail_url": item.thumbnail_url,
                "detail_url": detail.detail_url,
                "image_urls": detail.image_urls,
            }

            if not args.dry_run:
                with meta_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.download and not args.dry_run:
                goods_dir = out_dir / "images" / _safe_filename(item.goods_cd)
                goods_dir.mkdir(parents=True, exist_ok=True)

                # Save a small per-item metadata file for convenience.
                meta_item_path = goods_dir / "meta.json"
                if not meta_item_path.exists():
                    meta_item_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

                for image_url in detail.image_urls:
                    filename = _safe_filename(image_url.rsplit("/", 1)[-1])
                    dest = goods_dir / filename
                    if dest.exists():
                        continue
                    try:
                        download_file(session, url=image_url, dest_path=dest, timeout_s=args.timeout)
                    except Exception as exc:
                        print(f"[warn] download failed goods_cd={item.goods_cd} url={image_url} err={exc}", file=sys.stderr)
                        continue
        if reached_limit:
            break

    if reached_limit:
        print(f"Reached --max-items={args.max_items}. Stopping.", file=sys.stderr)
    if not args.dry_run:
        print(f"Done. Wrote metadata: {meta_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
