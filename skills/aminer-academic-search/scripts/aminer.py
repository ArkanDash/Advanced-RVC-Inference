#!/usr/bin/env python3
"""Dispatcher for the 27 AMiner Open Platform functions exposed via the
z-ai gateway's /v1/functions/invoke endpoint.

Each action maps 1:1 to an `aminer_*` function registered in the gateway.
Arguments are collected via argparse and forwarded as the `arguments` field.
Upstream response is printed as formatted JSON.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any, Callable, Tuple

REQUEST_TIMEOUT = 60
CONFIG_PATHS = [
    pathlib.Path.cwd() / ".z-ai-config",
    pathlib.Path.home() / ".z-ai-config",
    pathlib.Path("/etc/.z-ai-config"),
]


# ---------------------------------------------------------------------------
# Action registry
#
# Each entry defines one AMiner Open Platform API:
#   function_name: matches the key in handlers/aminer_open.go's AminerOpenSpecs
#   args:          (flag, dest, kind, help) — kind is "str"/"int"/"float"/
#                  "bool"/"json" (parsed from JSON for list/dict/nested values)
#
# This list is the *single source of truth* for what params each action takes,
# so the model only has to read one structure to learn the whole surface.
# ---------------------------------------------------------------------------

_Arg = Tuple[str, str, str, str]  # (flag, dest, kind, help)

ACTIONS: dict[str, dict[str, Any]] = {
    # ── Paper ────────────────────────────────────────────────────────────
    "paper_search": {
        "function": "aminer_paper_search",
        "help": "Locate a paper_id by (partial) title.",
        "args": [
            ("--title", "title", "str", "Paper title keyword (required)"),
            ("--page", "page", "int", "Page number, default 1"),
            ("--size", "size", "int", "Page size, default 10"),
        ],
    },
    "paper_search_pro": {
        "function": "aminer_paper_search_pro",
        "help": "Multi-condition paper search (author/org/venue/keyword).",
        "args": [
            ("--title", "title", "str", "Title keyword"),
            ("--keyword", "keyword", "str", "Subject keyword"),
            ("--abstract", "abstract", "str", "Abstract keyword"),
            ("--author", "author", "str", "Author name"),
            ("--org", "org", "str", "Organization name"),
            ("--venue", "venue", "str", "Venue name"),
            ("--order", "order", "str", "Sort: citation | year"),
            ("--page", "page", "int", "Page number, default 0"),
            ("--size", "size", "int", "Page size, default 10"),
        ],
    },
    "paper_qa_search": {
        "function": "aminer_paper_qa_search",
        "help": "AI Q&A-style search. 'query' and 'topic_*' are mutually exclusive.",
        "args": [
            ("--query", "query", "str", "Natural language question"),
            ("--use-topic", "use_topic", "bool", "Use topic_* fields instead of query"),
            ("--topic-high", "topic_high", "str", "High-level topic"),
            ("--topic-middle", "topic_middle", "str", "Mid-level topic"),
            ("--topic-low", "topic_low", "str", "Low-level topic"),
            ("--title", "title", "json", "List of titles (JSON array)"),
            ("--doi", "doi", "str", "DOI"),
            ("--year", "year", "json", "Year filter, e.g. [2020,2024]"),
            ("--sci-flag", "sci_flag", "bool", "Restrict to SCI venues"),
            ("--n-citation-flag", "n_citation_flag", "bool", "Return citation count"),
            ("--force-citation-sort", "force_citation_sort", "bool", "Sort by citations"),
            ("--force-year-sort", "force_year_sort", "bool", "Sort by year"),
            ("--author-terms", "author_terms", "json", "Author name list"),
            ("--org-terms", "org_terms", "json", "Org name list"),
            ("--author-id", "author_id", "json", "Author ID list"),
            ("--org-id", "org_id", "json", "Org ID list"),
            ("--venue-ids", "venue_ids", "json", "Venue ID list"),
            ("--size", "size", "int", "Page size, default 10"),
            ("--offset", "offset", "int", "Offset, default 0"),
        ],
    },
    "paper_info": {
        "function": "aminer_paper_info",
        "help": "Batch retrieve papers by ID list.",
        "args": [
            ("--ids", "ids", "json", "JSON array of paper_id strings (required)"),
        ],
    },
    "paper_detail": {
        "function": "aminer_paper_detail",
        "help": "Full paper info for a single paper_id.",
        "args": [
            ("--id", "id", "str", "paper_id (required)"),
        ],
    },
    "paper_relation": {
        "function": "aminer_paper_relation",
        "help": "Citation chain (cited papers) for a paper_id.",
        "args": [
            ("--id", "id", "str", "paper_id (required)"),
        ],
    },
    "paper_list_by_keywords": {
        "function": "aminer_paper_list_by_keywords",
        "help": "Batch keyword retrieval returning abstracts + metadata.",
        "args": [
            ("--keywords", "keywords", "json", "JSON array of keyword strings (required)"),
            ("--page", "page", "int", "Page number, default 0"),
            ("--size", "size", "int", "Page size, default 10"),
        ],
    },
    "paper_detail_by_condition": {
        "function": "aminer_paper_detail_by_condition",
        "help": "Year + venue dimension lookup. Year + venue_id both required.",
        "args": [
            ("--year", "year", "int", "Year (required)"),
            ("--venue-id", "venue_id", "str", "Venue ID (required)"),
        ],
    },
    # ── Scholar ──────────────────────────────────────────────────────────
    "person_search": {
        "function": "aminer_person_search",
        "help": "Search scholars by name and/or org.",
        "args": [
            ("--name", "name", "str", "Scholar name"),
            ("--org", "org", "str", "Organization name"),
            ("--org-id", "org_id", "json", "List of org IDs"),
            ("--offset", "offset", "int", "Offset, default 0"),
            ("--size", "size", "int", "Page size, default 5"),
        ],
    },
    "person_detail": {
        "function": "aminer_person_detail",
        "help": "Full scholar profile (bio/education/honors).",
        "args": [("--id", "id", "str", "person_id (required)")],
    },
    "person_figure": {
        "function": "aminer_person_figure",
        "help": "Scholar portrait (interests, work history).",
        "args": [("--id", "id", "str", "person_id (required)")],
    },
    "person_paper_relation": {
        "function": "aminer_person_paper_relation",
        "help": "List of papers by this scholar.",
        "args": [("--id", "id", "str", "person_id (required)")],
    },
    "person_patent_relation": {
        "function": "aminer_person_patent_relation",
        "help": "List of patents by this scholar.",
        "args": [("--id", "id", "str", "person_id (required)")],
    },
    "person_project": {
        "function": "aminer_person_project",
        "help": "Research projects (funding, dates, source).",
        "args": [("--id", "id", "str", "person_id (required)")],
    },
    # ── Organization ─────────────────────────────────────────────────────
    "org_search": {
        "function": "aminer_org_search",
        "help": "Search institutions by name keyword.",
        "args": [("--orgs", "orgs", "json", "JSON array of org name strings (required)")],
    },
    "org_detail": {
        "function": "aminer_org_detail",
        "help": "Org details by ID list.",
        "args": [("--ids", "ids", "json", "JSON array of org_id strings (required)")],
    },
    "org_person_relation": {
        "function": "aminer_org_person_relation",
        "help": "Affiliated scholars (10 per call).",
        "args": [
            ("--org-id", "org_id", "str", "org_id (required)"),
            ("--offset", "offset", "int", "Offset, default 0"),
        ],
    },
    "org_paper_relation": {
        "function": "aminer_org_paper_relation",
        "help": "Papers authored by org members (10 per call).",
        "args": [
            ("--org-id", "org_id", "str", "org_id (required)"),
            ("--offset", "offset", "int", "Offset, default 0"),
        ],
    },
    "org_patent_relation": {
        "function": "aminer_org_patent_relation",
        "help": "Org patent list (max page_size 10,000).",
        "args": [
            ("--id", "id", "str", "org_id (required)"),
            ("--page", "page", "int", "Page number, default 1"),
            ("--page-size", "page_size", "int", "Page size, default 100"),
        ],
    },
    "org_disambiguate": {
        "function": "aminer_org_disambiguate",
        "help": "Normalize raw org string.",
        "args": [("--org", "org", "str", "Raw org string (required)")],
    },
    "org_disambiguate_pro": {
        "function": "aminer_org_disambiguate_pro",
        "help": "Extract primary/secondary org IDs.",
        "args": [("--org", "org", "str", "Raw org string (required)")],
    },
    # ── Venue ────────────────────────────────────────────────────────────
    "venue_search": {
        "function": "aminer_venue_search",
        "help": "Search journals/conferences by name.",
        "args": [("--name", "name", "str", "Venue name (required)")],
    },
    "venue_detail": {
        "function": "aminer_venue_detail",
        "help": "Venue details (ISSN, abbreviation, type).",
        "args": [("--id", "id", "str", "venue_id (required)")],
    },
    "venue_paper_relation": {
        "function": "aminer_venue_paper_relation",
        "help": "Papers published in a venue, optionally filtered by year.",
        "args": [
            ("--id", "id", "str", "venue_id (required)"),
            ("--offset", "offset", "int", "Offset, default 0"),
            ("--limit", "limit", "int", "Limit, default 20"),
            ("--year", "year", "int", "Publication year"),
        ],
    },
    # ── Patent ───────────────────────────────────────────────────────────
    "patent_search": {
        "function": "aminer_patent_search",
        "help": "Search patents by name/keyword.",
        "args": [
            ("--query", "query", "str", "Search query (required)"),
            ("--page", "page", "int", "Page number, default 0"),
            ("--size", "size", "int", "Page size, default 10"),
        ],
    },
    "patent_info": {
        "function": "aminer_patent_info",
        "help": "Basic patent info.",
        "args": [("--id", "id", "str", "patent_id (required)")],
    },
    "patent_detail": {
        "function": "aminer_patent_detail",
        "help": "Full patent details (abstract, IPC, claims).",
        "args": [("--id", "id", "str", "patent_id (required)")],
    },
}


# ---------------------------------------------------------------------------
# Config loading and HTTP
# ---------------------------------------------------------------------------


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
                    f"warning: {path} has no 'token' field; the gateway requires "
                    "X-Token for auth and will return 401.",
                    file=sys.stderr,
                )
            return cfg
    raise SystemExit(
        "z-ai config not found. Create .z-ai-config in cwd, $HOME, or /etc "
        "with {\"baseUrl\": ..., \"apiKey\": ..., \"token\": ...}."
    )


def invoke(config: dict[str, Any], function_name: str, arguments: dict[str, Any]) -> Any:
    base_url = str(config["baseUrl"]).rstrip("/")
    url = f"{base_url}/functions/invoke"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "aminer-academic-search-skill/1.0",
        "Authorization": f"Bearer {config['apiKey']}",
        "X-Z-AI-From": "Z",
    }
    if config.get("token"):
        headers["X-Token"] = str(config["token"])
    if config.get("chatId"):
        headers["X-Chat-Id"] = str(config["chatId"])
    if config.get("userId"):
        headers["X-User-Id"] = str(config["userId"])

    body = json.dumps(
        {"function_name": function_name, "arguments": arguments},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:  # nosec B310
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
        raise SystemExit(f"gateway unreachable: {exc.reason}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid json response: {exc}") from exc

    if isinstance(parsed.get("error"), str) and parsed["error"]:
        raise SystemExit(f"gateway error: {parsed['error']}")
    return parsed.get("result")


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _cast(kind: str) -> Callable[[str], Any]:
    if kind == "int":
        return int
    if kind == "float":
        return float
    if kind == "json":
        return lambda s: json.loads(s)
    return str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call one of 27 AMiner Open Platform APIs via the z-ai gateway.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action", required=True, metavar="ACTION")
    for name, meta in ACTIONS.items():
        p = sub.add_parser(
            name,
            help=meta["help"],
            description=f"{meta['help']}  function_name: {meta['function']}",
        )
        for flag, dest, kind, help_text in meta["args"]:
            if kind == "bool":
                p.add_argument(flag, dest=dest, action="store_true", help=help_text)
            else:
                p.add_argument(flag, dest=dest, type=_cast(kind), help=help_text)
    return parser


def collect_arguments(ns: argparse.Namespace, spec: list[_Arg]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for _flag, dest, kind, _help in spec:
        value = getattr(ns, dest, None)
        if value is None:
            continue
        if kind == "bool" and value is False:
            continue
        args[dest] = value
    return args


def main() -> int:
    parser = build_parser()
    ns = parser.parse_args()

    meta = ACTIONS[ns.action]
    arguments = collect_arguments(ns, meta["args"])

    config = load_config()
    result = invoke(config, meta["function"], arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
