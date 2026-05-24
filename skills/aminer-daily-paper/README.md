# aminer-daily-paper

Skill for AMiner academic paper recommendations.

## Configuration

```bash
# TODO: fill in the actual API URL
export AMINER_API_URL="https://<your-api-host>/..."
```

## Usage

Natural language or explicit command:

```
/aminer-dp topics: multimodal agents, tool-use size: 5
recommend me recent papers on RAG
```

## Manual invocation

```bash
python3 scripts/recommend.py --topic "RAG" --topic "multimodal" --size 5 --language-sort zh
```

Output is a Markdown-formatted paper list.
