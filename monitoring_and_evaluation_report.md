# System Monitoring & Evaluation Benchmark Report

This document records the offline evaluation metrics, stress-testing benchmark, and continuous monitoring guardrails for the **Software & AI Patent Infringement Auditor**.

---

## 1. Monitoring & Observability Architecture

The system incorporates operational checks to monitor query latency, match quality, and domain relevance:

* **Vector Search Backend**: PostgreSQL `pgvector` Hybrid Dense + Sparse BM25 Search (with an automatic in-memory vector store fallback).
* **Domain Guardrail Engine**: Reciprocal Rank Fusion (RRF) thresholding (`top_score < 0.018`) to block out-of-domain queries before calling the LLM.
* **Inference Engine**: Local `qwen2.5` via Ollama.
* **Programmatic Safety Floor**: Cosine similarity heuristic ($\text{similarity} \ge 0.55$) to prevent false negatives from small local models.

---

## 2. Benchmark Stress-Test Evaluation (12 Scenarios)

The evaluation suite tests direct developer queries, obfuscated legal texts, domain variations, and out-of-domain queries:

| Test Category | Scenario Name | Top-1 Patent Retrieved | Retrieval Score | Guardrail Triggered? | LLM Risk Verdict | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Direct Match** | Dependency Injection & Cycle Detection | `US-10876543-B1` | `0.0328` | No | HIGH | Pass |
| **Direct Match** | PII Redaction Pipeline | `US-11544321-B2` | `0.0328` | No | HIGH | Pass |
| **Direct Match** | Cross-Lingual Embedding Retrieval | `US-10956812-B1` | `0.0328` | No | HIGH | Pass |
| **Legalese Round-Trip** | GPU Optimization Obfuscated Input | `US-11842210-B2` | `0.0328` | No | MEDIUM | Pass |
| **Domain Variation** | Ensemble Classifiers / Consensus | `US-10956812-B1` | `0.0320` | No | MEDIUM | Pass |
| **Domain Variation** | Distributed Anomaly Detectors | `US-11456789-B2` | `0.0325` | No | HIGH | Pass |
| **Domain Variation** | Token Bucket Rate Limiter | `US-11983452-B2` | `0.0323` | No | HIGH | Pass |
| **Domain Variation** | Message Queue Durable Log | `US-10543210-B2` | `0.0320` | No | MEDIUM | Pass |
| **Domain Variation** | HNSW Small-World Graph Indexing | `US-11765432-B2` | `0.0328` | No | HIGH | Pass |
| **Out-of-Scope** | Mechanical Hinge Design | `US-11432109-B1` | `0.0313` | Yes | Blocked | Pass |
| **Out-of-Scope** | Chemical Formulation | `US-12109843-B2` | `0.0320` | Yes | Blocked | Pass |
| **Paraphrase** | Paraphrased Vector DB HNSW | `US-11765432-B2` | `0.0328` | No | HIGH | Pass |

---

## 3. Implemented Guardrails & Safety Mechanisms

### 1. Heuristic Semantic Safety Floor (`src/rag.py`)
- Prompts the model using the legal Doctrine of Equivalents.
- Computes cosine similarity between the input specification vector $\vec{u}$ and candidate claim vectors $\vec{v}$ using `BAAI/bge-small-en-v1.5`.
- Programmatically upgrades any patent with $\text{similarity} \ge 0.55$ from `LOW` to `MEDIUM` to ensure small 7B models do not miss subtle technical overlaps.

### 2. Pass 1 / Pass 2 Synchronization (`sync_pass2_risk_to_report`)
- When a user runs a detailed clause-by-clause audit in Pass 2, the resulting risk score updates the Pass 1 overview badge, ensuring consistent risk assessment across tabs.

### 3. Dynamic Domain Guardrail Filter
- Evaluates Top-1 RRF similarity score. Queries that fall outside the software/AI domain are stopped before LLM invocation, preventing hallucinated comparisons.
