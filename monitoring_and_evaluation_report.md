# System Monitoring & Evaluation Benchmark Report

This document records the offline evaluation metrics, 4-pillar continuous monitoring framework, benchmark stress tests, and production observability architecture for the **Software & AI Patent Infringement Auditor**.

---

## 1. The 4 Pillars of LLM & RAG Monitoring

In traditional software, monitoring checks server uptime and HTTP response codes. In an **LLM/RAG pipeline**, queries can return an `HTTP 200 OK` while producing inaccurate classifications or severe latency degradation. 

Our application continuously observes **4 critical operational pillars** via `src/monitoring.py`:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    THE 4 PILLARS OF LLM/RAG MONITORING                    │
├───────────────────────────────┬────────────────────────────────────────────┤
│ 1. Retrieval Quality & Drift  │ 2. Generation & Safety Guardrails          │
│    - Top-1 RRF similarity     │    - Semantic Safety Floor activation rate │
│    - Data & vocabulary drift  │    - Out-of-domain guardrail rejections    │
├───────────────────────────────┼────────────────────────────────────────────┤
│ 3. System Latency & Resources │ 4. User Feedback (Human-in-the-Loop)       │
│    - DB vs. LLM generation   │    - Thumbs Up/Down satisfaction rating    │
│    - Pass 1 vs. Pass 2 timing │    - Design-around actionability feedback  │
└───────────────────────────────┴────────────────────────────────────────────┘
```

### Pillar 1: Retrieval Quality & Semantic Drift Monitoring
- **Tracked Metric**: Top-1 Reciprocal Rank Fusion (RRF) Retrieval Score per transaction.
- **Purpose**: A sudden downward shift in average RRF scores signals that users are entering architectures outside the currently indexed vector space, indicating that new USPTO patent batches must be ingested via the `dlt` pipeline.

### Pillar 2: Generation & Safety Guardrails Monitoring
- **Tracked Metric**: Cosine Safety Floor Trigger Rate ($\text{similarity} \ge 0.55$) and Out-of-Scope Filter Rate.
- **Purpose**: Quantifies how often the deterministic embedding floor overrides small 7B model false negatives. A high activation rate signals the need to refine system prompt examples.

### Pillar 3: System Latency & Performance Monitoring
- **Tracked Metric**: End-to-end execution breakdown (Vector/FTS Database Latency, Pass 1 Screening Latency, Pass 2 Deep Audit Latency).
- **Purpose**: Ensures query response times remain within acceptable engineering thresholds on local hardware.

### Pillar 4: Human-in-the-Loop User Feedback
- **Tracked Metric**: Thumbs Up (`👍 Helpful`) vs. Thumbs Down (`👎 Inaccurate`) ratings submitted on claim translation cards.
- **Purpose**: Provides real-world ground-truth signal from software engineers to guide ongoing prompt engineering.

---

## 2. Interactive Monitoring Dashboard (Streamlit Tab 4)

All telemetry is aggregated in real-time in the Streamlit web interface across **5 distinct visualizations**:

| Dashboard Visualization | Tracked Telemetry | Strategic Engineering Utility |
| :--- | :--- | :--- |
| **1. Latency Breakdown** | PostgreSQL search vs. Pass 1 LLM generation | Pinpoints compute bottlenecks between retrieval and model inference. |
| **2. Risk Verdict Distribution** | Breakdown of `HIGH`, `MEDIUM`, and `LOW` verdicts | Detects systemic bias or conservative skew in risk classifications. |
| **3. Semantic Drift Trend** | Top-1 RRF score over time | Detects domain drift when new engineering specifications are queried. |
| **4. Safety Floor Activation** | Frequency of programmatic risk upgrades | Measures the protective impact of the heuristic cosine floor. |
| **5. User Feedback Rating** | Positive vs. Negative human feedback ratio | Direct feedback loop on patent translation quality and design-around advice. |

---

## 3. Benchmark Stress-Test Evaluation (12 Scenarios)

The evaluation suite validates retrieval performance, domain guardrail triggers, and risk classifications across 12 distinct scenarios:

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

## 4. Implemented Guardrails & Safety Mechanisms

### 1. Heuristic Semantic Safety Floor (`src/rag.py`)
- Prompts the model using the legal Doctrine of Equivalents.
- Computes cosine similarity between the input specification vector $\vec{u}$ and candidate claim vectors $\vec{v}$ using `BAAI/bge-small-en-v1.5`.
- Programmatically upgrades any patent with $\text{similarity} \ge 0.55$ from `LOW` to `MEDIUM` to ensure small 7B models do not miss subtle technical overlaps.

### 2. Pass 1 / Pass 2 Synchronization (`sync_pass2_risk_to_report`)
- When a user runs a detailed clause-by-clause audit in Pass 2, the resulting risk score updates the Pass 1 overview badge, ensuring consistent risk assessment across tabs.

### 3. Dynamic Domain Guardrail Filter
- Evaluates Top-1 RRF similarity score. Queries that fall outside the software/AI domain are stopped before LLM invocation, preventing hallucinated comparisons.
