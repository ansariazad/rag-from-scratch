# RAG from Scratch

**rag-from-scratch** is a full-spectrum Retrieval-Augmented Generation (RAG) pipeline built entirely from first principles in pure Python — no LangChain, no LlamaIndex, no vector database SDKs. Implements Parts 1–18 of Lance Martin's RAG course: document chunking, TF-IDF vectors, Multi-Query RRF fusion, Corrective RAG (CRAG), Self-RAG, and semantic routing.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![No Deps](https://img.shields.io/badge/Core-Zero_Dependencies-brightgreen)
![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Recall](https://img.shields.io/badge/CRAG+RRF_Recall-100%25-brightgreen)

---

## What is this?

A production-grade RAG system coded from scratch. Every retrieval and generation technique is hand-implemented:

- **TF-IDF / n-gram vectors** with L2 normalization — no FAISS, no Pinecone
- **Reciprocal Rank Fusion (RRF)** — `score = Σ 1 / (k + rank_i)` fused across multiple query variants
- **Corrective RAG (CRAG)** — grades chunks for relevance, falls back to global search if scores are too low
- **Self-RAG** — model decides when retrieval is needed and self-critiques its own answers
- **Semantic routing** — classifies the query domain and dispatches to the right knowledge index

Pluggable LLM backend: uses Ollama locally when `OLLAMA_MODEL` is set, falls back to a deterministic offline synthesis engine so it always works without any API key.

---

## Key Features

- **Multi-Query Fan-Out** — expands one ambiguous query into 3 perspective variants (lexical, conceptual, contrastive) for broader coverage
- **Reciprocal Rank Fusion (RRF)** — fuses ranked retrieval lists from all query variants: `1 / (k + rank)`, de-duplicates by chunk ID
- **Semantic Routing** — routes queries to `transformers_db`, `rag_db`, `agents_mcp_db`, or `evals_db` based on keyword intent matching
- **Corrective RAG (CRAG)** — scores every retrieved chunk for relevance, discards below-threshold chunks, triggers global fallback search when needed
- **Self-RAG** — after answering, verifies the response against retrieved context, flags hallucinations with confidence estimates
- **Faithfulness Scoring** — measures how many answer tokens are grounded in source chunks (scored 0.974 in evals)
- **Pluggable Ollama backend** — set `OLLAMA_MODEL=llama3.2` to route synthesis through a local LLM; offline fallback always works

---

## Architecture

```
                        User Query
                            │
                    ┌───────▼───────┐
                    │ Semantic Router│  → picks domain index
                    └───────┬───────┘
                            │
              ┌─────────────▼──────────────┐
              │     Multi-Query Expansion   │
              │  Q1 (lexical)              │
              │  Q2 (conceptual)           │
              │  Q3 (contrastive)          │
              └─────┬──────┬──────┬────────┘
                    │      │      │
              ┌─────▼──────▼──────▼────────┐
              │  TF-IDF Retrieval (per Q)   │
              │  Top-K chunks per query     │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Reciprocal Rank Fusion     │
              │  score = Σ 1/(k + rank_i)  │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  CRAG Relevance Grading     │
              │  Pass? → Generate           │
              │  Fail? → Global Fallback    │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  LLM Synthesis              │
              │  (Ollama or offline engine) │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  Self-RAG Verification      │
              │  Faithfulness Scoring       │
              └─────────────┬──────────────┘
                            │
                       Final Answer
```

---

## Prerequisites

**Python 3.11+** required. The core pipeline has **zero external dependencies** — runs with stdlib only.

### Optional: Ollama (for real LLM synthesis)
```bash
# macOS
brew install ollama
ollama serve &
ollama pull llama3.2

# Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

### Optional: httpx (for async requests)
```bash
pip install httpx
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ansariazad/rag-from-scratch.git
cd rag-from-scratch

# 2. Run offline (zero dependencies)
python3 02_rag_from_scratch.py

# 3. Run with real Ollama LLM
OLLAMA_MODEL=llama3.2 python3 02_rag_from_scratch.py
```

---

## Usage

### Offline Mode (default)
```bash
python3 02_rag_from_scratch.py
```

**Expected output:**
```
=== RAG PIPELINE DEMO ===
Query: "How does CRAG improve retrieval quality?"
Router → domain: rag_db

[Multi-Query] Generated 3 query variants
[RRF Fusion ] Merged 9 candidate chunks → top 3

[CRAG] Chunk relevance scores: [0.72, 0.81, 0.68]
[CRAG] All chunks above threshold — proceeding to generation

[Self-RAG] Answer faithfulness: 0.974
Answer: Corrective RAG grades retrieved documents for relevance...

=== A/B COMPARISON ===
System A (Naive)  Recall: 1/4 (25%)
System B (CRAG+RRF) Recall: 4/4 (100%)  ← +75pp improvement
```

### With Ollama LLM
```bash
OLLAMA_MODEL=llama3.2 python3 02_rag_from_scratch.py
# LLM synthesis will use local llama3.2 for answers
```

---

## Knowledge Base

The built-in corpus covers 5 AI engineering domains:

| Domain | Topics Covered |
|--------|---------------|
| `transformers_db` | GPT architecture, pre-training, alignment (SFT, RLHF) |
| `rag_db` | Multi-Query RRF, CRAG, Self-RAG, faithfulness scoring |
| `agents_mcp_db` | Anthropic agent patterns, MCP protocol, ACI tools |
| `evals_db` | LLM-as-judge, TPR/TNR calibration, A/B pipelines |

---

## Eval Results

Tested by [`ai-evals-suite`](https://github.com/ansariazad/ai-evals-suite):

| Metric | System A (Naive) | System B (CRAG+RRF) | Δ |
|--------|-----------------|--------------------|----|
| Recall | 25% (1/4) | **100% (4/4)** | **+75pp** |
| Faithfulness | — | **0.974** | — |

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OLLAMA_MODEL` | Local Ollama model to use for synthesis | `llama3.2` |

Copy `.env.example` → `.env`:
```bash
cp .env.example .env
```

---

## File Structure

```
rag-from-scratch/
├── 02_rag_from_scratch.py  # Full RAG pipeline (chunker, TF-IDF, RRF, CRAG, Self-RAG)
├── setup_and_run_mac.sh    # macOS one-shot setup
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── .gitignore
└── README.md
```

---

## References

- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020
- [CRAG: Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) — Yan et al., 2024
- [Self-RAG](https://arxiv.org/abs/2310.11511) — Asai et al., 2023
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Cormack et al., 2009
- [Learn RAG From Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8) — Lance Martin (LangChain / freeCodeCamp)

---

## Contributing

Pull requests welcome! Suggested additions: BM25 retriever, dense embedding retrieval (sentence-transformers), streaming LLM synthesis.

---

## License

MIT © 2026

---

## Author

**Azad Ansari** — [GitHub](https://github.com/ansariazad) | [AutoShorts](https://github.com/ansariazad/AutoShorts)
