# 📚 RAG from Scratch — Full Retrieval-Augmented Generation Pipeline

Build a production-grade RAG system from scratch — Multi-Query RRF, Corrective RAG (CRAG), and Self-RAG without frameworks.

## ✨ Features

- **Multi-Query Retrieval** — generates 3 query variants, fuses results with Reciprocal Rank Fusion (RRF)
- **Corrective RAG (CRAG)** — grades retrieved docs, falls back to web search if irrelevant
- **Self-RAG** — model decides when to retrieve and critiques its own output
- **Faithfulness scoring** — citation verification against source chunks (scored 0.974 in evals)
- **Ollama integration** — uses `llama3.2` locally, offline fallback always works

## 🏗️ Architecture

```
Query → Multi-Query Expansion → Parallel Retrieval
      ↓
    RRF Fusion → Top-K Chunks → Relevance Grading
      ↓
  Relevant? → YES → Generate Answer → Citation Check
            → NO  → Web Search Fallback → Re-retrieve
```

## 🚀 Quick Start

```bash
pip install torch numpy httpx

# Offline mode (deterministic)
python 02_rag_from_scratch.py

# With Ollama (real LLM)
OLLAMA_MODEL=llama3.2 python 02_rag_from_scratch.py
```

## 📊 Eval Results

| Metric | Score |
|--------|-------|
| System B (CRAG+RRF) Recall | 100% |
| System A (baseline) Recall | 25% |
| Improvement | +75pp |
| Faithfulness | 0.974 |

## 🔗 References

- [RAG Paper](https://arxiv.org/abs/2005.11401) — Lewis et al.
- [CRAG](https://arxiv.org/abs/2401.15884) — Yan et al.
- [Self-RAG](https://arxiv.org/abs/2310.11511) — Asai et al.
- [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Cormack et al.

## 📄 License

MIT
