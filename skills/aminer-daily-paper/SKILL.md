---
name: aminer-daily-paper
description: "Get personalized academic paper recommendations. Activate whenever the user asks for paper recommendations — explicit command (/aminer-dp) or natural language (e.g. 'recommend me papers on RAG', 'suggest recent papers on multimodal agents'). Workflow: extract topics / author / aminer_author_id from the input, invoke scripts/recommend.py, return results as Markdown."
---

# aminer-daily-paper

Personalized academic paper recommendation.

## When to activate

Any time the user asks for paper recommendations:
- Explicit command: `/aminer-dp`, `/aminer-dp topics: RAG, multimodal agents`
- Natural language: `recommend me recent papers on multimodal agents`, `suggest some papers on tool-use`, `I work on RAG, give me a few papers`

## Input parsing (done by the model)

Before calling the script, extract from the user input:

| Field | Description |
|-------|-------------|
| `topics` | Research topics, 1–3 closely related terms work best |
| `author_name` | Scholar name |
| `author_org` | Scholar institution (improves disambiguation) |
| `aminer_author_id` | AMiner scholar ID (24-char hex) |
| `size` | Number of papers, default 5, max 20 |
| `language_sort` | `zh` or `en`, optional |

At least one of `topics` / `author_name` / `aminer_author_id` should be provided.

## Call strategy

| Scenario | Strategy |
|----------|----------|
| Single topic or scholar | 1 call, `size=5` |
| User specifies a number | 1 call, honor the number (max 20) |
| Multiple distinct topics | 1 call per topic group, `size=3–5` each, ~15 papers total |
| Broad request with no topics | 1 call, `size=5` |

## Execution

The script reads `.z-ai-config` (JSON) following the `z-ai-web-dev-sdk`
convention, searching in this order:
1. `./.z-ai-config` (cwd)
2. `~/.z-ai-config` (home)
3. `/etc/.z-ai-config` (system)

Required fields: `baseUrl`, `apiKey`, `token` (JWT for `X-Token`).
Optional: `chatId`, `userId`.

```bash
python3 "{baseDir}/scripts/recommend.py" \
  [--topic "multimodal agents"] \
  [--topic "tool-use"] \
  [--author-name "Jie Tang"] \
  [--author-org "Tsinghua University"] \
  [--aminer-author-id "696259801cb939bc391d3a37"] \
  [--size 5] \
  [--language-sort zh]
```

The script POSTs to `${baseUrl}/functions/invoke` with headers
`Authorization: Bearer ${apiKey}`, `X-Z-AI-From: Z`, and `X-Token: ${token}`
(plus optional `X-Chat-Id` / `X-User-Id`), and prints the paper list as
Markdown.

## Output handling

- On success, the script prints a Markdown paper list — forward it directly to the user.
- On non-zero exit, stderr contains the error — relay it concisely to the user.

## Error handling

- API returns an error → relay the error; do not switch to another skill.
- No results → suggest the user broaden topics or adjust the query.
