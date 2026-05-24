#!/usr/bin/env python3
"""Fetch AMiner paper recommendations and render results as Markdown."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_SIZE = 5
MAX_SIZE = 20
REQUEST_TIMEOUT = 30
CONFIG_PATHS = [
    pathlib.Path.cwd() / ".z-ai-config",
    pathlib.Path.home() / ".z-ai-config",
    pathlib.Path("/etc/.z-ai-config"),
]


def load_config() -> dict[str, Any]:
    for path in CONFIG_PATHS:
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"failed to read {path}: {exc}") from exc
        if cfg.get("baseUrl") and cfg.get("apiKey"):
            if not cfg.get("token"):
                print(
                    f"warning: {path} has no 'token' field; the API requires "
                    "X-Token for auth and will return 401.",
                    file=sys.stderr,
                )
            return cfg
    raise SystemExit(
        "z-ai config not found. Create .z-ai-config in cwd, $HOME, or /etc "
        "with {\"baseUrl\": ..., \"apiKey\": ..., \"token\": ...}."
    )


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _truncate(text: str, limit: int) -> str:
    text = _clean(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    topics = [t for t in (args.topic or []) if _clean(t)]
    if topics:
        arguments["topics"] = topics
    if args.author_name:
        arguments["author_name"] = args.author_name
    if args.author_org:
        arguments["author_org"] = args.author_org
    if args.aminer_author_id:
        arguments["aminer_author_id"] = args.aminer_author_id
    if args.language_sort in {"zh", "en"}:
        arguments["language_sort"] = args.language_sort
    if args.start_year:
        arguments["start_year"] = args.start_year
    if args.end_year:
        arguments["end_year"] = args.end_year

    size = args.size if args.size else DEFAULT_SIZE
    size = max(1, min(size, MAX_SIZE))
    arguments["size"] = size

    return {"function_name": "aminer_recommend", "arguments": arguments}


def call_api(config: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    base_url = str(config["baseUrl"]).rstrip("/")
    url = f"{base_url}/functions/invoke"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "aminer-daily-paper-skill/1.0",
        "Authorization": f"Bearer {config['apiKey']}",
        "X-Z-AI-From": "Z",
    }
    if config.get("token"):
        headers["X-Token"] = str(config["token"])
    if config.get("chatId"):
        headers["X-Chat-Id"] = str(config["chatId"])
    if config.get("userId"):
        headers["X-User-Id"] = str(config["userId"])

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as resp:  # nosec B310
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        hint = ""
        if exc.code == 401 and "X-Token" in detail:
            hint = " (hint: add a valid 'token' field to .z-ai-config)"
        elif exc.code == 403:
            hint = " (hint: request rejected by auth — check 'apiKey' / X-Z-AI-From)"
        raise SystemExit(f"http {exc.code}: {detail or exc.reason}{hint}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"api unreachable: {exc.reason}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid json response: {exc}") from exc

    if isinstance(parsed.get("error"), str) and parsed["error"]:
        raise SystemExit(f"api error: {parsed['error']}")

    result = parsed.get("result")
    if not isinstance(result, list):
        raise SystemExit(f"unexpected response shape: {raw[:300]}")
    return [p for p in result if isinstance(p, dict)]


def render_markdown(papers: list[dict[str, Any]], topics: list[str]) -> str:
    if not papers:
        return "No papers returned. Try broadening the topics or adjusting the query."

    lines: list[str] = []
    header = f"Recommended {len(papers)} paper(s)"
    if topics:
        header += f" (topics: {' / '.join(topics[:5])})"
    lines.append(header)

    for idx, paper in enumerate(papers, start=1):
        lines.append("")
        lines.append("---")
        lines.append("")
        title = _clean(paper.get("title"))
        links = paper.get("links") if isinstance(paper.get("links"), dict) else {}
        url = _clean(paper.get("paper_url") or links.get("aminer") or links.get("arxiv"))
        title_line = f"**{idx}. [{title}]({url})**" if url else f"**{idx}. {title}**"
        lines.append(title_line)

        meta: list[str] = []
        year = paper.get("year")
        if year:
            meta.append(f"Year: {year}")
        keywords = paper.get("keywords") or []
        if keywords:
            meta.append("Keywords: " + " / ".join(str(k) for k in keywords[:5]))
        if meta:
            lines.append(" | ".join(meta))

        authors = paper.get("authors") or []
        if authors:
            author_str = ", ".join(str(a) for a in authors[:6])
            if len(authors) > 6:
                author_str += " et al."
            lines.append(f"Authors: {author_str}")

        summary = _clean(paper.get("summary"))
        if summary:
            lines.append("")
            lines.append(_truncate(summary, 300))

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch AMiner paper recommendations.")
    parser.add_argument("--topic", action="append", default=[], help="research topic (repeatable)")
    parser.add_argument("--author-name", default="")
    parser.add_argument("--author-org", default="")
    parser.add_argument("--aminer-author-id", default="")
    parser.add_argument("--size", type=int, default=0)
    parser.add_argument("--language-sort", default="", choices=["", "zh", "en"])
    parser.add_argument("--start-year", type=int, default=0)
    parser.add_argument("--end-year", type=int, default=0)
    args = parser.parse_args()

    payload = build_payload(args)
    config = load_config()
    papers = call_api(config, payload)
    topics = payload["arguments"].get("topics") or []
    print(render_markdown(papers, topics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
