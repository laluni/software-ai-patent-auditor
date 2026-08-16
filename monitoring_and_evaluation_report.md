# System Monitoring and Evaluation Benchmark Report

This document records the offline evaluation metrics, 4-pillar continuous monitoring framework, benchmark stress tests, and production observability architecture for the Software & AI Patent Infringement Auditor.

---

## 1. The 4 Pillars of LLM and RAG Monitoring

In traditional software, monitoring tracks server uptime, CPU load, and HTTP response codes. In an LLM and RAG pipeline, transactions can return an HTTP 200 status while suffering from retrieval degradation, out-of-domain drift, guardrail intervention spikes, or ungrounded outputs.

The application implements continuous telemetry across 4 operational pillars in `src/monitoring.py`:

```
+----------------------------------------------------------------------------+
|                    THE 4 PILLARS OF LLM/RAG MONITORING                    |
+-------------------------------+--------------------------------------------+
| 1. Retrieval Quality & Drift  | 2. Generation & Safety Guardrails          |
|    - Top-1 RRF similarity     |    - Semantic Safety Floor activation rate |
|    - Windowed drift detection |    - Out-of-scope guardrail classification |
|    - Auto re-ingestion alerts |    - Request-level risk tracking           |
+-------------------------------+--------------------------------------------+
| 3. System Latency & Resources | 4. User Feedback (Human-in-the-Loop)       |
|    - PostgreSQL vs. LLM timing|    - Per-claim helpful/inaccurate ratings  |
|    - Pass 1 vs. Pass 2 timing |    - Shared request_id traceability        |
|    - Millisecond precision    |    - Qualitative commentary logging        |
+-------------------------------+--------------------------------------------+
```

### Pillar 1: Retrieval Quality and Semantic Drift Monitoring
- **Tracked Metric**: Top-1 Reciprocal Rank Fusion (RRF) score per query.
- **Drift Detection Window**: Evaluates a rolling window of 5 consecutive in-domain queries. If scores remain below the 0.020 threshold, the system flags sustained semantic drift and raises a re-ingestion alert.
- **Out-of-Scope Protection**: Blocked out-of-scope queries are explicitly excluded from the drift window to prevent false re-ingestion triggers.

### Pillar 2: Generation and Safety Guardrails Monitoring
- **Tracked Metric**: Semantic Safety Floor trigger rate (cosine similarity >= 0.55) and Domain Guardrail classification.
- **Guardrail Separation**: Distinguishes between `SIMILARITY_FLOOR` (programmatic upgrade preventing false negatives) and `OUT_OF_SCOPE` (domain boundary rejection) events.
- **Risk Distribution**: Logs overall infringement risk classifications (`HIGH`, `MEDIUM`, `LOW`, `BLOCKED`) for historical analysis.

### Pillar 3: System Latency and Compute Breakdown
- **Tracked Metric**: Stage-specific latency breakdown (`db_latency_ms`, `pass1_latency_ms`, `pass2_latency_ms`, `total_latency_ms`).
- **Pass 2 Disaggregation**: Pass 2 is an optional on-demand operation; its latency is updated via `update_pass2_latency()` by `request_id` to ensure baseline Pass 1 screening metrics are not artificially inflated.

### Pillar 4: Human-in-the-Loop Feedback Attribution
- **Tracked Metric**: Per-claim user ratings (+1 for Helpful, -1 for Needs Improvement) with optional text notes.
- **Traceability**: All feedback events contain `request_id`, `patent_id`, and `claim_index`, establishing direct cross-pillar linkage to the specific query and retrieval scores that produced the claim translation.
- **Data Integrity**: Zero-feedback states report `None` (rendered as `N/A`) rather than default 100% values.

---

## 2. Interactive Monitoring Dashboard (Streamlit Tab 4)

Telemetry is persisted to `data/query_monitoring_logs.json` and `data/user_feedback_logs.json`, providing historical analytics across application sessions.

| Dashboard Visualization | Tracked Telemetry | Strategic Utility |
| :--- | :--- | :--- |
| **1. Latency Breakdown** | PostgreSQL search vs. Pass 1 LLM generation (sec) | Identifies compute bottlenecks between vector indexing and model inference. |
| **2. Risk Verdict Distribution** | Breakdown of HIGH, MEDIUM, LOW, and BLOCKED verdicts | Detects classification drift and conservative skew in model auditing. |
| **3. Semantic Drift Trend** | Top-1 RRF score over time | Observes vector similarity health across queries. |
| **4. Safety Floor Activation** | Frequency of programmatic risk upgrades vs. standard assessments | Measures the real-world coverage of the 0.55 cosine safety net. |
| **5. User Feedback Rating** | Helpful vs. Needs Improvement distribution | Tracks practical satisfaction with generated legal translations and design-around advice. |

---

## 3. Benchmark Stress-Test Evaluation (12 Scenarios)

The evaluation suite validates retrieval accuracy, domain guardrails, and risk classifications across 12 structured scenarios:

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

## 4. Implemented Guardrails and Safety Mechanisms

### 1. Heuristic Semantic Safety Floor (`src/rag.py`)
- Evaluates functional equivalence under the Doctrine of Equivalents.
- Calculates dense cosine similarity between specification vectors and retrieved claim vectors using `BAAI/bge-small-en-v1.5`.
- Automatically upgrades patents with similarity >= 0.55 from `LOW` to `MEDIUM` risk, mitigating false negatives in local models.

### 2. Pass 1 / Pass 2 Synchronization (`sync_pass2_risk_to_report`)
- Deep audits performed in Pass 2 synchronize back to the Pass 1 overall report and status badge, maintaining consistent risk representation.

### 3. Dynamic Domain Guardrail Filter
- Validates query domain boundaries before executing LLM inference, preventing ungrounded comparisons on non-software topics.
