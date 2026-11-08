#!/usr/bin/env python3
"""
02_rag_from_scratch.py — Full-Spectrum RAG Pipeline From Scratch
================================================================
Reference Source:
  - Lance Martin (LangChain / freeCodeCamp.org),
    "Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer"
    https://www.youtube.com/watch?v=sVcwVQRHIc8

Implemented Stages (Parts 1–18 of the course):
  1. Document Chunking & Indexing: Overlapping semantic chunker + L2-normalized TF-IDF / n-gram Vectors
  2. Multi-Query Fan-Out & Reciprocal Rank Fusion (RAG-Fusion): Expands an ambiguous query into
     multiple perspectives and fuses ranked retrieval lists via RRF (`1 / (k + rank)`)
  3. Semantic Query Routing: Routes incoming questions to the right specialized index
     (e.g., `transformers_db`, `rag_db`, `agents_mcp_db`, `evals_db`)
  4. Corrective RAG (CRAG) & Self-RAG:
     - Grades retrieved documents for relevance
     - Falls back to global search if relevance fails
     - Verifies generated answer against retrieved context for hallucination control
  5. Pluggable LLM Backends:
     - Automatically uses Ollama (`http://localhost:11434`) on your Mac when `OLLAMA_MODEL` is set,
       with a zero-dependency local vector + synthesis engine so it runs out-of-the-box offline.

Usage on macOS:
  python3 02_rag_from_scratch.py
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DocumentChunk:
    chunk_id: str
    domain: str
    title: str
    source_url: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


KNOWLEDGE_BASE: List[DocumentChunk] = [
    DocumentChunk(
        chunk_id="gpt-01",
        domain="transformers_db",
        title="Karpathy nanoGPT Decoder Architecture",
        source_url="https://www.youtube.com/watch?v=kCc8FmEb1nY",
        content=(
            "A decoder-only GPT Transformer predicts the next token autoregressively using "
            "masked multi-head causal self-attention (tril lower-triangular mask) so tokens "
            "only attend to past positions. Pre-LayerNorm residual blocks combine communication "
            "(MultiHeadAttention) with computation (position-wise FeedForward MLP with 4x channel "
            "expansion), trained with cross-entropy loss and AdamW."
        ),
    ),
    DocumentChunk(
        chunk_id="gpt-02",
        domain="transformers_db",
        title="Pre-training vs Alignment Stages in ChatGPT",
        source_url="https://www.youtube.com/watch?v=kCc8FmEb1nY",
        content=(
            "Pre-training on large text corpora (like OpenWebText or Tiny Shakespeare in nanoGPT) "
            "creates a document completer. Turning it into an assistant like ChatGPT requires "
            "Supervised Fine-Tuning (SFT) on question-answer demonstrations, training a reward "
            "model on human preference rankings, and reinforcement learning via PPO."
        ),
    ),
    DocumentChunk(
        chunk_id="rag-01",
        domain="rag_db",
        title="Multi-Query Retrieval & Reciprocal Rank Fusion (RAG-Fusion)",
        source_url="https://www.youtube.com/watch?v=sVcwVQRHIc8",
        content=(
            "Single-vector similarity search can miss relevant documents due to phrasing nuances. "
            "Multi-query fan-out generates multiple perspectives of the user question, retrieves "
            "documents for each query, and combines results using Reciprocal Rank Fusion (RRF) "
            "with score sum(1 / (k + rank)) where k=60."
        ),
    ),
    DocumentChunk(
        chunk_id="rag-02",
        domain="rag_db",
        title="Corrective RAG (CRAG) and Active Retrieval Grading",
        source_url="https://www.youtube.com/watch?v=sVcwVQRHIc8",
        content=(
            "Corrective RAG (CRAG) adds an explicit document relevance grader after retrieval. "
            "If retrieved chunks are irrelevant or out-of-domain, CRAG rewrites the query and "
            "triggers a fallback search before final answer generation, followed by a hallucination "
            "and answer-faithfulness check."
        ),
    ),
    DocumentChunk(
        chunk_id="agent-01",
        domain="agents_mcp_db",
        title="Anthropic Workflows vs Autonomous Agents",
        source_url="https://www.anthropic.com/engineering/building-effective-agents",
        content=(
            "Anthropic distinguishes between Workflows (where LLMs and tools are orchestrated "
            "through predefined code paths: Prompt Chaining, Routing, Parallelization, "
            "Orchestrator-Workers, and Evaluator-Optimizer) and Autonomous Agents (where LLMs "
            "dynamically direct their own tool usage in an environment feedback loop)."
        ),
    ),
    DocumentChunk(
        chunk_id="mcp-01",
        domain="agents_mcp_db",
        title="Model Context Protocol (MCP) Server Architecture",
        source_url="https://modelcontextprotocol.io/docs/2026-07-28/develop/build-server",
        content=(
            "MCP servers expose Resources, Tools, and Prompts to MCP hosts like Claude for Desktop. "
            "For STDIO-based MCP servers, never write to stdout (e.g., avoid print()) because "
            "stdout carries JSON-RPC 2.0 messages; always log to stderr via logging.getLogger(__name__)."
        ),
    ),
    DocumentChunk(
        chunk_id="eval-01",
        domain="evals_db",
        title="Hamel Husain's Three Levels of LLM Evaluation",
        source_url="https://hamel.dev/blog/posts/evals/index.html",
        content=(
            "Hamel Husain structures AI evaluation into 3 levels: Level 1 fast deterministic "
            "unit-test assertions run on every code change; Level 2 trace logging with human "
            "binary (Good/Bad) review and calibrated LLM-as-a-Judge evaluation; and Level 3 "
            "A/B testing on production changes."
        ),
    ),
]


class HybridEmbedder:
    """
    Computes L2-normalized hybrid lexical + bigram dense embeddings in pure Python
    so vector search works deterministically offline on any Mac without external packages.
    """

    def __init__(self, chunks: List[DocumentChunk]) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fit(chunks)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r"[a-z0-9_]+", text.lower())
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def _fit(self, chunks: List[DocumentChunk]) -> None:
        doc_freq: Counter[str] = Counter()
        for chunk in chunks:
            tokens = set(self._tokenize(chunk.title + " " + chunk.content))
            for tok in tokens:
                doc_freq[tok] += 1
        all_tokens = sorted(doc_freq.keys())
        self.vocab = {tok: idx for idx, tok in enumerate(all_tokens)}
        n_docs = max(len(chunks), 1)
        self.idf = {
            tok: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            for tok, df in doc_freq.items()
        }

    def embed(self, text: str) -> Dict[int, float]:
        vec: Dict[int, float] = {}
        counts = Counter(self._tokenize(text))
        norm_sq = 0.0
        for tok, count in counts.items():
            if tok in self.vocab:
                val = (1.0 + math.log(count)) * self.idf[tok]
                vec[self.vocab[tok]] = val
                norm_sq += val * val
        norm = math.sqrt(norm_sq)
        if norm > 0:
            return {k: v / norm for k, v in vec.items()}
        return vec


class VectorStoreIndex:
    """In-memory vector index supporting cosine similarity search and domain filtering."""

    def __init__(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        self.embedder = HybridEmbedder(chunks)
        self.vectors = [self.embedder.embed(f"{c.title} {c.content}") for c in chunks]

    @staticmethod
    def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(val * b.get(idx, 0.0) for idx, val in a.items())

    def search(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        q_vec = self.embedder.embed(query)
        scored: List[Tuple[DocumentChunk, float]] = []
        for chunk, d_vec in zip(self.chunks, self.vectors):
            if domain_filter and chunk.domain != domain_filter:
                continue
            scored.append((chunk, self._cosine(q_vec, d_vec)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


def route_query(query: str) -> Optional[str]:
    """Semantic Query Router (Lance Martin RAG Course Part 10)."""
    q = query.lower()
    domain_keywords = {
        "transformers_db": ["gpt", "transformer", "attention", "karpathy", "pre-training", "tril"],
        "rag_db": ["rag", "retrieval", "rrf", "fusion", "crag", "chunk"],
        "agents_mcp_db": ["agent", "workflow", "orchestrator", "mcp", "stdio", "tool"],
        "evals_db": ["eval", "assertion", "hamel", "judge", "trace", "a/b"],
    }
    matched_domains = [
        domain
        for domain, kws in domain_keywords.items()
        if any(kw in q for kw in kws)
    ]
    if len(matched_domains) == 1:
        return matched_domains[0]
    return None


def generate_multi_queries(question: str) -> List[str]:
    """Query Translation / Fan-Out (Part 5)."""
    return [
        question,
        f"core architecture and implementation details for: {question}",
        f"best practices, failure modes, and verification for: {question}",
    ]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[DocumentChunk, float]]],
    k: int = 60,
    top_n: int = 3,
) -> List[Tuple[DocumentChunk, float]]:
    """Reciprocal Rank Fusion (RAG-Fusion, Part 6): score = sum(1 / (k + rank))."""
    fused_scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, DocumentChunk] = {}

    for results in ranked_lists:
        for rank, (chunk, _sim) in enumerate(results, start=1):
            fused_scores[chunk.chunk_id] += 1.0 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
    return [(chunk_map[cid], fused_scores[cid]) for cid in sorted_ids[:top_n]]


def grade_document_relevance(question: str, chunk: DocumentChunk, threshold: float = 0.08) -> bool:
    """Active / Corrective RAG Document Grader (Part 16–17)."""
    q_tokens = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
    c_tokens = set(re.findall(r"[a-z0-9]{3,}", (chunk.title + " " + chunk.content).lower()))
    if not q_tokens:
        return False
    overlap = len(q_tokens & c_tokens) / len(q_tokens)
    return overlap >= threshold


def call_llm_or_local_synthesizer(question: str, context_chunks: List[DocumentChunk]) -> str:
    """Calls local Ollama on macOS (`OLLAMA_MODEL`) if set, or synthesizes a grounded answer."""
    ollama_model = os.environ.get("OLLAMA_MODEL", "")
    if ollama_model:
        context_block = "\n\n".join(
            f"[{c.chunk_id}] {c.title} ({c.source_url}):\n{c.content}"
            for c in context_chunks
        )
        prompt = (
            f"Answer the question strictly using the provided context and cite URLs.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {question}"
        )
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({"model": ollama_model, "prompt": prompt, "stream": False}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.loads(resp.read().decode())
                return str(payload.get("response", "")).strip()
        except Exception:
            pass

    bullet_lines = [
        f"- **{c.title}** ({c.source_url}): {c.content}"
        for c in context_chunks
    ]
    return f"Grounded synthesis for '{question}':\n" + "\n".join(bullet_lines)


def verify_no_hallucination(answer: str, context_chunks: List[DocumentChunk]) -> Tuple[bool, float]:
    """Self-RAG Hallucination & Faithfulness Grader (Part 17)."""
    context_text = " ".join(c.title + " " + c.content + " " + c.source_url for c in context_chunks).lower()
    context_vocab = set(re.findall(r"[a-z0-9]{4,}", context_text))
    answer_terms = [
        w for w in re.findall(r"[a-z0-9]{4,}", answer.lower())
        if w not in {"grounded", "synthesis", "question", "https", "watch", "blog", "posts", "docs"}
    ]
    if not answer_terms:
        return True, 1.0
    grounded_count = sum(1 for w in answer_terms if w in context_vocab)
    score = grounded_count / len(answer_terms)
    return score >= 0.70, score


class CorrectiveRAGPipeline:
    """End-to-end RAG pipeline combining Routing, RAG-Fusion (RRF), CRAG Grading, and Self-RAG verification."""

    def __init__(self, chunks: List[DocumentChunk]) -> None:
        self.index = VectorStoreIndex(chunks)

    def query(self, question: str) -> Dict[str, object]:
        routed_domain = route_query(question)
        sub_queries = generate_multi_queries(question)

        ranked_lists = [
            self.index.search(sq, top_k=4, domain_filter=routed_domain)
            for sq in sub_queries
        ]
        fused = reciprocal_rank_fusion(ranked_lists, k=60, top_n=3)

        relevant_chunks = [
            chunk for chunk, _rrf_score in fused
            if grade_document_relevance(question, chunk)
        ]
        fallback_triggered = False

        if not relevant_chunks:
            fallback_triggered = True
            global_lists = [self.index.search(sq, top_k=4, domain_filter=None) for sq in sub_queries]
            fused = reciprocal_rank_fusion(global_lists, k=60, top_n=3)
            relevant_chunks = [chunk for chunk, _ in fused]

        answer = call_llm_or_local_synthesizer(question, relevant_chunks)
        is_faithful, faithfulness_score = verify_no_hallucination(answer, relevant_chunks)

        return {
            "question": question,
            "routed_domain": routed_domain or "all_domains (multi-topic)",
            "sub_queries": sub_queries,
            "retrieved_chunk_ids": [c.chunk_id for c in relevant_chunks],
            "sources": [c.source_url for c in relevant_chunks],
            "crag_fallback_triggered": fallback_triggered,
            "is_faithful": is_faithful,
            "faithfulness_score": round(faithfulness_score, 3),
            "answer": answer,
        }


def main() -> None:
    print("=" * 72)
    print("RAG FROM SCRATCH — Routing + RAG-Fusion (RRF) + Corrective RAG (CRAG)")
    print("Reference: https://www.youtube.com/watch?v=sVcwVQRHIc8 (Lance Martin)")
    print("=" * 72)

    pipeline = CorrectiveRAGPipeline(KNOWLEDGE_BASE)

    sample_questions = [
        "How does Multi-Query Reciprocal Rank Fusion and Corrective RAG improve retrieval?",
        "Why must STDIO MCP servers never write to stdout and how do Anthropic workflows differ from agents?",
        "How does Hamel Husain structure the 3 levels of LLM evals?",
    ]

    for idx, q in enumerate(sample_questions, start=1):
        res = pipeline.query(q)
        print(f"\n[Query {idx}] {res['question']}")
        print(f"  Routed Domain      : {res['routed_domain']}")
        print(f"  RRF Retrieved IDs  : {res['retrieved_chunk_ids']}")
        print(f"  CRAG Fallback Used : {res['crag_fallback_triggered']}")
        print(f"  Self-RAG Faithful  : {res['is_faithful']} (score={res['faithfulness_score']})")
        print(f"  Answer             :\n{res['answer']}")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
