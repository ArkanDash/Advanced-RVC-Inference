---
name: aminer-academic-search
description: >
  ACADEMIC PRIORITY: Activate whenever the user's query involves academic,
  scholarly, or research-related topics — papers, citations, scholars,
  institutions, venues, patents, research trends, or any "who published what /
  where / when" question. Takes precedence over general web search for academic
  data needs. Routes through the z-ai gateway's `/v1/functions/invoke` endpoint
  to the AMiner Open Platform (27 APIs, 5 workflows).
---

# aminer-academic-search

Wraps 27 AMiner Open Platform APIs through the local gateway. The gateway owns
the `AMINER_API_KEY`; the skill only needs a `.z-ai-config` file to reach the
gateway.

## When to activate

Any academic/scholarly query — papers, citations, scholars, institutions,
venues, patents. Covers:
- Scholar full profile (bio, education, honors, papers, patents, projects)
- Paper deep dive (full abstract, keywords, authors, citation chain)
- Multi-condition or semantic paper search
- Institution research capability analysis
- Venue annual monitoring
- Patent deep details (IPC/CPC, assignee, claims)

## Config

Script reads `.z-ai-config`, in order:
1. `./.z-ai-config`
2. `~/.z-ai-config`
3. `/etc/.z-ai-config`

Required fields: `baseUrl`, `apiKey`, `token` (JWT). Optional: `chatId`,
`userId`.

## Invocation

```bash
python3 "{baseDir}/scripts/aminer.py" <ACTION> [--flag value …]
```

The script POSTs to `${baseUrl}/functions/invoke` with
`Authorization: Bearer ${apiKey}`, `X-Z-AI-From: Z`, `X-Token: ${token}`,
and prints the upstream JSON result (pretty-formatted).

Run `python3 scripts/aminer.py --help` to list actions, or
`python3 scripts/aminer.py <ACTION> --help` for per-action flags.

### Examples

```bash
# Locate paper_id by title
python3 scripts/aminer.py paper_search --title "Attention is all you need"

# Locate scholar by name + org
python3 scripts/aminer.py person_search --name "Jie Tang" --org "Tsinghua"

# Full paper details
python3 scripts/aminer.py paper_detail --id 57a4e91aac44365e35c98a1e

# Natural-language Q&A search
python3 scripts/aminer.py paper_qa_search --query "retrieval-augmented generation" --size 5

# Venue papers for a given year (body with JSON list param)
python3 scripts/aminer.py venue_paper_relation --id "53e9ba63b7602d9702f08c5d" --year 2024 --limit 10
```

JSON-typed flags (`--title`, `--ids`, `--keywords`, `--year` on qa_search, etc.)
expect a JSON-encoded value, e.g. `--ids '["abc","def"]'`.

## 27 APIs (quick reference)

| Action | Method | Purpose |
|---|---|---|
| paper_search | GET | Locate paper_id by title |
| paper_search_pro | GET | Multi-condition paper search |
| paper_qa_search | POST | Natural-language / topic Q&A |
| paper_info | POST | Batch paper info by IDs |
| paper_detail | GET | Full paper details |
| paper_relation | GET | Citation chain |
| paper_list_by_keywords | GET | Batch thematic retrieval |
| paper_detail_by_condition | GET | Year + venue dimension |
| person_search | POST | Locate person_id |
| person_detail | GET | Scholar bio/education/honors |
| person_figure | GET | Interests + work history |
| person_paper_relation | GET | Scholar's papers |
| person_patent_relation | GET | Scholar's patents |
| person_project | GET | Funded projects |
| org_search | POST | Locate org by name |
| org_detail | POST | Org description/type |
| org_person_relation | GET | Affiliated scholars |
| org_paper_relation | GET | Org papers |
| org_patent_relation | GET | Org patents |
| org_disambiguate | POST | Normalize org string |
| org_disambiguate_pro | POST | Extract org IDs |
| venue_search | POST | Locate venue_id |
| venue_detail | POST | ISSN/type/abbreviation |
| venue_paper_relation | POST | Papers by venue (+ year filter) |
| patent_search | POST | Patent keyword search |
| patent_info | GET | Basic patent info |
| patent_detail | GET | Full patent details (IPC/claims) |

## 5 Workflows (orchestrate via multiple calls)

### Scholar Profile
```
person_search → person_detail
              → person_figure
              → person_paper_relation
              → person_patent_relation
              → person_project
```

### Paper Deep Dive
```
paper_search → paper_detail → paper_relation → paper_info
```
If `paper_search` is empty, fall back to `paper_search_pro`.

### Org Analysis
```
org_disambiguate_pro → org_detail
                     → org_person_relation
                     → org_paper_relation
                     → org_patent_relation
```
If `org_disambiguate_pro` returns no ID, fall back to `org_search`.

### Venue Papers
```
venue_search → venue_detail (optional) → venue_paper_relation
```

### Patent Analysis
```
patent_search → patent_info / patent_detail
```

## Error handling

- `warning: .z-ai-config has no 'token' field` → add `token` to config.
- `http 401 … missing X-Token header (hint: …)` → same.
- `http 403` → gateway auth rejected; check `apiKey` / `X-Z-AI-From`.
- `gateway error: aminer_* failed: http 4xx …` → upstream refused; check API
  key on the gateway side (`AMINER_API_KEY` env var) or parameter shape.

## Entity URL templates

- Paper: `https://www.aminer.cn/pub/{paper_id}`
- Scholar: `https://www.aminer.cn/profile/{scholar_id}`
- Patent: `https://www.aminer.cn/patent/{patent_id}`
- Journal: `https://www.aminer.cn/open/journal/detail/{journal_id}`

Always append the relevant URL when presenting entities to the user.
