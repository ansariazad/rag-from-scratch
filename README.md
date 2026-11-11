# 📚 RAG from Scratch — Full RAG Pipeline in Pure Python, Zero Frameworks

> **No LangChain. No LlamaIndex. No vector DB SDK. Just Python.**

A production-grade Retrieval-Augmented Generation system built entirely from first principles — TF-IDF vectors, Reciprocal Rank Fusion, Corrective RAG, and Self-RAG, all in one file. Covers **Parts 1–18** of Lance Martin's RAG course. CRAG+RRF achieves **100% recall** vs 25% for naive retrieval.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Zero Deps](https://img.shields.io/badge/Core-Zero_Dependencies-22c55e)](https://github.com/ansariazad/rag-from-scratch)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-F97316)](https://ollama.ai)
[![Recall](https://img.shields.io/badge/CRAG+RRF_Recall-100%25-22c55e)](https://github.com/ansariazad/ai-evals-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ansariazad/rag-from-scratch?style=social)](https://github.com/ansariazad/rag-from-scratch/stargazers)

---

## 📺 Demo

```
$ python3 02_rag_from_scratch.py

━━━ QUERY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"How does Corrective RAG improve retrieval quality?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Router   ] → rag_db

[MultiQ   ] Generated 3 query variants:
            Q1: "corrective RAG relevance grading"
            Q2: "CRAG fallback web search mechanism"
            Q3: "retrieval quality improvement techniques"

[RRF      ] Fused 9 candidates → top 3 chunks
            Score: chunk-rag-01=0.312  chunk-rag-02=0.287  chunk-rag-03=0.198

[CRAG     ] Relevance grades: [0.81, 0.74, 0.69] — all above threshold ✓

[Synth    ] Generating answer...

[Self-RAG ] Faithfulness: 0.974 ✓

━━━ ANSWER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Corrective RAG grades each retrieved chunk for relevance
before passing it to the LLM. If chunks score below the
threshold, it triggers a global fallback search to find
better context, eliminating hallucinations from bad retrieval.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ A/B COMPARISON ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
System A — Naive single-query RAG   : Recall  25% (1/4)
System B — CRAG + Multi-Query RRF   : Recall 100% (4/4)  🏆 +75pp
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📑 Table of Contents

- [Why This Exists](#-why-this-exists)
- [Techniques Implemented](#-techniques-implemented)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Knowledge Base](#-knowledge-base)
- [Eval Results](#-eval-results)
- [Roadmap](#-roadmap)
- [References](#-references)

---

## 💡 Why This Exists

Most RAG tutorials use LangChain or LlamaIndex as a black box. When things break in production, you have no idea why. This repo exposes every retrieval decision:

| Component | Here | LangChain | LlamaIndex |
|-----------|------|-----------|------------|
| TF-IDF vectors | Hand-coded, L2-normalized | Wrapped | Wrapped |
| RRF fusion | Explicit `1/(k+rank)` formula | Hidden | Hidden |
| CRAG relevance grader | Transparent scoring | Plugin | Plugin |
| Self-RAG verification | Custom faithfulness fn | Not included | Not included |
| External dependencies | **0** (stdlib only) | 50+ | 30+ |
| Debuggable | ✅ every step printed | ❌ | ❌ |

---

## 🛠️ Techniques Implemented

| # | Technique | What It Does |
|---|-----------|-------------|
| 1 | **Document Chunking** | Overlapping semantic chunks with metadata (domain, source URL, chunk ID) |
| 2 | **TF-IDF Vectors** | L2-normalized term frequency × inverse document frequency scoring |
| 3 | **Multi-Query Fan-Out** | Expands 1 query into 3 variants (lexical, conceptual, contrastive) |
| 4 | **Reciprocal Rank Fusion** | `score = Σ 1/(k + rank_i)` across all query variants |
| 5 | **Semantic Routing** | Keyword-intent classifier routes to the right knowledge index |
| 6 | **Corrective RAG (CRAG)** | Grades chunk relevance, triggers global fallback if scores fail |
| 7 | **Self-RAG** | Post-generation faithfulness verification, hallucination flagging |
| 8 | **Ollama Integration** | `OLLAMA_MODEL=llama3.2` routes synthesis through local LLM |

---

## 🏗️ Architecture

```
                       User Query
                           │
                   ┌───────▼────────┐
                   │ Semantic Router │ → picks domain index
                   └───────┬────────┘
                           │
           ┌───────────────▼───────────────┐
           │      Multi-Query Expansion     │
           │  Q1 (lexical variant)          │
           │  Q2 (conceptual variant)       │
           │  Q3 (contrastive variant)      │
           └────┬──────────┬────────┬──────┘
                │          │        │
           ┌────▼──────────▼────────▼──────┐
           │   TF-IDF Retrieval (per query) │
           │   Top-K chunks scored per Q    │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │    Reciprocal Rank Fusion      │
           │    score = Σ 1 / (k + rank_i) │
           │    De-duplicate by chunk_id    │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │    CRAG Relevance Grading      │
           │    Pass threshold? → Generate  │
           │    Fail?  → Global Fallback    │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │    LLM Synthesis               │
           │    Ollama (if set) or offline  │
           └───────────────┬───────────────┘
                           │
           ┌───────────────▼───────────────┐
           │    Self-RAG Verification       │
           │    Faithfulness = grounded     │
           │    tokens / total tokens       │
           └───────────────┬───────────────┘
                           │
                      Final Answer
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/ansariazad/rag-from-scratch.git
cd rag-from-scratch

# No install needed — zero dependencies!
python3 02_rag_from_scratch.py

# With real Ollama LLM synthesis
brew install ollama && ollama pull llama3.2
OLLAMA_MODEL=llama3.2 python3 02_rag_from_scratch.py
```

---

## 📖 Usage

### Offline (zero dependencies)
```bash
python3 02_rag_from_scratch.py
```

### With Ollama
```bash
OLLAMA_MODEL=llama3.2 python3 02_rag_from_scratch.py
```

### Environment
```bash
cp .env.example .env
# Set OLLAMA_MODEL=llama3.2 in .env
```

---

## 🗄️ Knowledge Base

Built-in corpus covering 5 AI engineering domains:

| Domain | Topics |
|--------|--------|
| `transformers_db` | GPT architecture, pre-training, SFT, RLHF, alignment |
| `rag_db` | Multi-Query RRF, CRAG, Self-RAG, faithfulness scoring |
| `agents_mcp_db` | Anthropic agent patterns, MCP protocol, ACI tool design |
| `evals_db` | LLM-as-judge, TPR/TNR calibration, A/B eval pipelines |
| `mcp_db` | JSON-RPC 2.0, MCP STDIO transport, Claude Desktop config |

---

## 🧪 Eval Results

Verified by [`ai-evals-suite`](https://github.com/ansariazad/ai-evals-suite):

| Metric | System A (Naive RAG) | System B (CRAG + RRF) | Delta |
|--------|---------------------|----------------------|-------|
| Recall | 25% (1/4) | **100% (4/4)** | **+75pp** 🏆 |
| Faithfulness | — | **0.974** | — |

---

## 🗺️ Roadmap

- [ ] Dense embeddings (sentence-transformers, no API key)
- [ ] BM25 retriever option
- [ ] Streaming LLM synthesis
- [ ] Persistent vector store (SQLite + numpy)
- [ ] Multi-document PDF ingestion

---

## 📚 References

| Paper / Resource | Authors |
|------------------|---------|
| [RAG for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401) | Lewis et al., 2020 |
| [CRAG: Corrective RAG](https://arxiv.org/abs/2401.15884) | Yan et al., 2024 |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | Asai et al., 2023 |
| [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Cormack et al., 2009 |
| [Learn RAG From Scratch (video)](https://www.youtube.com/watch?v=sVcwVQRHIc8) | Lance Martin |

---

## 🤝 Contributing

PRs welcome! Open an issue before major changes.

---

## ⭐ Support

If this demystified RAG for you, **star the repo** — it helps others discover it!

[![GitHub stars](https://img.shields.io/github/stars/ansariazad/rag-from-scratch?style=for-the-badge)](https://github.com/ansariazad/rag-from-scratch/stargazers)

---

## 📄 License

MIT © 2026

---

## 👤 Author

**Azad Ansari** — Python Developer & AI Engineer, Mumbai

[![GitHub](https://img.shields.io/badge/GitHub-ansariazad-181717?logo=github)](https://github.com/ansariazad)
[![Portfolio](https://img.shields.io/badge/Portfolio-ansariazad.github.io-blue)](https://ansariazad.github.io)
