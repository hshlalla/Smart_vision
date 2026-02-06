#!/usr/bin/env python3
"""
Merge multiple crawler outputs into one dataset folder (dedupe by goods_cd).

This expects the output of `dataset/crawler.py`:
  <root>/gparts_items.jsonl
  <root>/images/<goods_cd>/{meta.json, *.jpg, ...}

The merged output will be:
  <output>/gparts_items.jsonl
  <output>/images/<goods_cd>/...

Large crawled datasets should remain ignored by git via the repo `.gitignore`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Merge gparts crawler datasets and dedupe by goods_cd.")
    p.add_argument("--output", required=True, help="Output dataset directory.")
    p.add_argument("--inputs", nargs="+", required=True, help="Input dataset directories.")
    p.add_argument("--force", action="store_true", help="Allow reusing an existing output directory.")
    args = p.parse_args(argv)

    out_root = Path(args.output).expanduser().resolve()
    if out_root.exists() and not args.force:
        raise SystemExit(f"Output already exists: {out_root} (use --force to overwrite)")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)

    # Merge jsonl by goods_cd
    merged: dict[str, dict] = {}
    for in_dir in [Path(x).expanduser().resolve() for x in args.inputs]:
        for obj in _iter_jsonl(in_dir / "gparts_items.jsonl"):
            goods_cd = str(obj.get("goods_cd") or "").strip()
            if not goods_cd:
                continue
            if goods_cd not in merged:
                merged[goods_cd] = dict(obj)
                merged[goods_cd]["sources"] = [in_dir.name]
            else:
                merged[goods_cd]["sources"] = sorted(set((merged[goods_cd].get("sources") or []) + [in_dir.name]))
                # union image_urls
                a = set(merged[goods_cd].get("image_urls") or [])
                b = set(obj.get("image_urls") or [])
                merged[goods_cd]["image_urls"] = sorted(a | b)

    # Copy images, dedupe within goods_cd by sha
    for goods_cd in sorted(merged.keys()):
        dest_dir = out_root / "images" / goods_cd
        dest_dir.mkdir(parents=True, exist_ok=True)
        seen: set[str] = set()

        for in_dir in [Path(x).expanduser().resolve() for x in args.inputs]:
            src_dir = in_dir / "images" / goods_cd
            if not src_dir.exists():
                continue
            for fp in sorted(src_dir.iterdir()):
                if not fp.is_file():
                    continue
                if fp.name == "meta.json":
                    if (dest_dir / "meta.json").exists():
                        continue
                    shutil.copy2(fp, dest_dir / "meta.json")
                    continue
                if fp.suffix.lower() not in IMAGE_EXTS:
                    continue
                digest = _sha256(fp)
                if digest in seen:
                    continue
                seen.add(digest)
                dest = dest_dir / fp.name
                if dest.exists():
                    # name collision with different content
                    dest = dest_dir / f"{fp.stem}__{in_dir.name}{fp.suffix}"
                shutil.copy2(fp, dest)

    out_jsonl = out_root / "gparts_items.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for goods_cd in sorted(merged.keys()):
            obj = dict(merged[goods_cd])
            obj["goods_cd"] = goods_cd
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

