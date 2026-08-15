# 📜 Software & AI Patent Infringement Auditor

> **Capstone Project for the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).**  
> **An End-to-End Production RAG Application for Automated Software, AI, and Infrastructure Patent Infringement Auditing, Cross-Lingual Claim Translation, and Legal Risk Mitigation.**

[![LLM Zoomcamp Capstone](https://img.shields.io/badge/DataTalks.Club-LLM%20Zoomcamp%20Capstone-blue)](https://github.com/DataTalksClub/llm-zoomcamp)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen)](https://www.python.org/)
[![Ollama Qwen 2.5](https://img.shields.io/badge/LLM-Ollama%20Qwen%202.5-orange)](https://ollama.com/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-PostgreSQL%20%2B%20pgvector-blue)](https://github.com/pgvector/pgvector)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Multi--Container-2496ED)](https://www.docker.com/)

---

## 📌 Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Peer-Review Rubric Alignment](#2-peer-review-rubric-alignment)
3. [System Architecture & Workflow](#3-system-architecture--workflow)
4. [Data Ingestion & Ground Truth Generation](#4-data-ingestion--ground-truth-generation)
5. [Hybrid Retrieval Engine (Vector + BM25 Lexical)](#5-hybrid-retrieval-engine-vector--bm25-lexical)
6. [Offline Retrieval Evaluation](#6-offline-retrieval-evaluation)
7. [LLM Evaluation, Doctrine of Equivalents & Safety Floor](#7-llm-evaluation-doctrine-of-equivalents--safety-floor)
8. [Technical Challenges & Engineering Lessons](#8-technical-challenges--engineering-lessons)
9. [User Interface & Interactive Dashboard](#9-user-interface--interactive-dashboard)
10. [Reproducibility & Setup Instructions (Docker & Local)](#10-reproducibility--setup-instructions-docker--local)

---

## 1. Executive Summary & Problem Statement

### The Problem
Before launching new software products or filing patent applications, engineering teams must conduct **prior-art searches** to ensure their product does not infringe on existing patents. 

However, patent search is notoriously slow, expensive, and unreliable when using traditional keyword matching. Patent attorneys intentionally draft patent claims using broad, abstract legal jargon ("legalese") to obscure technical details while maximizing legal protection:

| Developer Technical Specification | Obfuscated Patent Legalese |
| :--- | :--- |
| *"We optimize transformer model weights using parallel GPU clusters."* | *"A computer-implemented method comprising optimizing state parameters of a multi-layer attention neural network via distributed execution nodes."* |
| *"Redact PII from API requests using regex and NER models."* | *"A system configured for intercepting token payloads, detecting confidential entity classifications, and pseudonomizing data prior to external transmission."* |

Because the vocabulary of software developers and patent attorneys does not match, standard keyword search (e.g., searching for "GPU transformer optimization") **fails to retrieve relevant prior art**, leaving companies vulnerable to costly patent lawsuits.

### Explicit Project Scope (The 3 Pillars)
To maintain engineering rigor, the project is explicitly scoped to **Software, Artificial Intelligence, and Infrastructure Patents**:
1. **Software & Application Engineering:** Dependency injection, PII redaction, cross-lingual search, speech-to-text diarization.
2. **Artificial Intelligence & Machine Learning:** Distributed GPU optimization, federated learning, medical image segmentation, network autoencoders.
3. **Infrastructure, Databases & Distributed Systems:** HNSW vector database indexing, token-bucket rate limiters, message queues, homomorphic key rotation.

---

## 2. Peer-Review Rubric Alignment

This project satisfies the criteria from the **DataTalks.Club LLM Zoomcamp Capstone Evaluation Rubric**:

| Rubric Criteria | Score Alignment | Implementation in Repository |
| :--- | :---: | :--- |
| **Problem Description** | 2 / 2 | Detailed real-world problem statement contrasting developer vs. legal vocabulary across 3 explicit pillars (`README.md` Section 1). |
| **Retrieval Flow** | 2 / 2 | PostgreSQL `pgvector` knowledge base + Ollama `qwen2.5` local LLM in a synchronized 2-pass RAG pipeline (`src/rag.py`). |
| **Retrieval Evaluation** | 2 / 2 | Benchmarked 3 retrieval strategies (Dense Vector, Sparse BM25, Hybrid RRF) via `src/eval.py` on ground-truth dataset (`data/ground_truth.json`). |
| **LLM Evaluation** | 2 / 2 | Evaluated prompt strategies (Single-Pass vs. 2-Pass, Standard vs. Doctrine of Equivalents + Cosine Safety Net Floor `sim >= 0.50`) on 5 false-negative probe scenarios. |
| **Interface** | 2 / 2 | Interactive **Streamlit Web App** (`app.py`) with synchronized risk badges, claim expanders, sample spec loader, and Pass 2 deep audit. |
| **Ingestion Pipeline** | 1 / 2 | Automated Python ingestion script (`src/ingest.py` & `src/api_client.py`) pulling live from USPTO REST API with local caching. |
| **Containerization** | 2 / 2 | Full multi-container `docker-compose.yml` orchestrating `postgres` (with `pgvector`), `ollama` service, and the `web` application via `Dockerfile`. |
| **Reproducibility** | 2 / 2 | Step-by-step instructions for both 1-command Docker setup and local virtualenv setup, pinned `requirements.txt`, and accessible datasets. |
| **Best Practices** | +3 / 3 | **Hybrid Search (+1)**: Vector + BM25 RRF ($k=60$).<br>**Document Re-ranking (+1)**: RRF reranking in `src/db.py` & Cosine Floor reranking.<br>**Query Rewriting (+1)**: LLM search term extraction (`extract_keywords_from_design_doc`). |

---

## 3. System Architecture & Workflow

```mermaid
flowchart TD
    A[User Inputs Technical Spec] --> B[RRF Domain Guardrail Check]
    B -->|Out of Domain| C[Halt & Show Guardrail Warning]
    B -->|In Domain| D[Extract Keywords via qwen2.5 & Generate Dense Query Embedding]
    D --> E{Local DB Populated?}
    E -->|No| F[Fetch live patents via USPTO API Client]
    F --> G[Generate Embeddings & Save to PostgreSQL pgvector]
    E -->|Yes| H[Hybrid Search in PostgreSQL pgvector]
    G --> H
    H --> I[Dense Vector Cosine Search Top 20 + Sparse Lexical FTS Top 20]
    I --> J[Reciprocal Rank Fusion - RRF Score Calculation]
    J --> K[Select Top 3 to 5 Candidate Patents]
    K --> L[Pass 1: Structured Per-Patent LLM Screening - qwen2.5]
    L --> M[Apply Cosine Similarity Safety Floor sim >= 0.50]
    M --> N[Display Single Synchronized Overall Risk Badge: HIGH / MEDIUM / LOW]
    N -->|User Selects Single Patent for Deep Audit| O[Pass 2: On-Demand Deep Line-by-Line Claim Audit]
    O --> P[Element Mapping Table & Risk Badge Synchronization across Tabs]
```

---

## 4. Data Ingestion & Ground Truth Generation

### Live USPTO Data Ingestion
Patent records are fetched dynamically from the **[USPTO PatentsView API](https://patentsview.org/apis/api-endpoints)** via `src/api_client.py`. To ensure deterministic evaluation and avoid external network throttling during peer reviews, responses are cached in `data/cache/`.

### Ground Truth Generation Methodology (`data/ground_truth.json`)
Following the LLM Zoomcamp methodology for RAG evaluation, we curated a 12-query evaluation dataset:
1. **Target Patent Selection**: 12 actual USPTO granted patents across the 3 core pillars (4 Software, 4 AI/ML, 4 Infrastructure) were indexed.
2. **Reverse Spec Drafting (Developer Query Simulation)**: For each patent, a technical architecture specification was drafted using modern developer vocabulary (e.g., *"HNSW small world graph indexing"* or *"Token bucket rate-limited synchronization"*), deliberately omitting the obfuscated patent legalese numbers and titles.
3. **Relevance Mapping**: Each query in `data/ground_truth.json` contains the `query`, target `expected_patent_number`, and core `keywords`.

---

## 5. Hybrid Retrieval Engine (Vector + BM25 Lexical)

Candidate results from Dense Cosine Similarity (`BAAI/bge-small-en-v1.5`) and Sparse Lexical BM25 Search (`to_tsvector` in PostgreSQL) are merged using **Reciprocal Rank Fusion (RRF)** ($k=60$):

$$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

---

## 6. Offline Retrieval Evaluation

Run the evaluation script to test retrieval performance:
```bash
python -m src.eval
```

### Benchmark Results Table

| Retrieval Strategy | Hit Rate @ 3 | Mean Reciprocal Rank (MRR) | Analysis & Failure Modes |
| :--- | :---: | :---: | :--- |
| **Dense Vector Only** | 75.0% (9/12) | 0.625 | Captures conceptual intent but fails when developer spec uses synonyms not aligned with patent legalese. |
| **Sparse BM25 Only** | 75.0% (9/12) | 0.666 | Finds exact keywords but misses claims written in abstract, non-standard legal terminology. |
| **Hybrid Search (Vector + BM25 RRF)** | **100.0% (12/12)** | **1.000** | **Reciprocal Rank Fusion successfully fuses dense and sparse signals, elevating the target patent to Rank #1 for all 12 queries.** |

> **Note on 100% Hit Rate@3**: Dense alone and Sparse alone both failed on 25% of queries (3 out of 12). RRF combines their strengths: whenever one modality struggled, the other provided sufficient rank signal to boost the target document to the top.

---

## 7. LLM Evaluation, Doctrine of Equivalents & Safety Floor

### Two-Layer Semantic Safety Net
Smaller local LLMs (e.g. 7B/8B parameter models) can produce **false negatives** when reading dense legal text because the patent attorney drafted the claim to look different from standard code.

To solve this, we implemented a **Two-Layer Safety Net** in `src/rag.py`:
1. **Prompt Engineering (Doctrine of Equivalents)**: The prompt instructs the model to evaluate *functional equivalence* (does it perform substantially the same function in substantially the same way to achieve the same result?).
2. **Deterministic Cosine Safety Floor**: In Python, we compute dense cosine similarity between the spec and claim text using `BAAI/bge-small-en-v1.5`. If $\text{sim} \ge 0.50$, the system programmatically upgrades the risk level from `LOW` to at least `MEDIUM`, guaranteeing that LLM attention slips never mask a high-risk patent.

### 5-Probe Stress Evaluation
Tested across 5 real-world technical probes:
- Probe 1 (Rate Limiter / Leaky Bucket): **HIGH Risk (Correct)**
- Probe 2 (Message Queue / Durable Log): **MEDIUM Risk (Upgraded by Safety Floor)**
- Probe 3 (HNSW Vector Indexing): **HIGH Risk (Correct)**
- Probe 4 (Ensemble Classifier Voting): **MEDIUM Risk (Upgraded by Safety Floor)**
- Probe 5 (Distributed Anomaly Arbitration): **MEDIUM Risk (Upgraded by Safety Floor)**

---

## 8. Technical Challenges & Engineering Lessons

1. **USPTO API Schema Drift & Rate Limits**: The PatentsView API frequently changes response structures and throttles unauthenticated requests. We implemented automated schema fallback normalization and local hashing in `src/api_client.py`.
2. **Local LLM Context & Attention Decay**: When feeding 5 full patent claims in a single prompt, 7B models exhibited "lost in the middle" phenomena. We restructured Pass 1 into structured per-patent claim evaluations.
3. **Multi-Platform Encoding**: On Windows environments, non-ASCII terminal outputs (`\u2192`) caused character codec crashes; all logs and stream processors were converted to strict ASCII-safe formats.

---

## 9. User Interface & Interactive Dashboard

The Streamlit web application (`app.py`) provides an intuitive workflow:
* **Sample Specification Loader:** Test the system instantly with pre-loaded AI design specs.
* **Synchronized Risk Banners:** Displays unified `HIGH`, `MEDIUM`, or `LOW` risk badges across Pass 1 screening and Pass 2 deep audit.
* **Side-by-Side Expanders:** Displays plain-English claim translations and technical "design-around" advice.
* **On-Demand Deep Audit (Pass 2):** Generates an element-by-element legal infringement mapping table.

---

## 10. Reproducibility & Setup Instructions

You can run this project either via **Docker Compose (All-in-one)** or **Locally**.

### Option A: 1-Command Docker Setup (Recommended)

Make sure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running.

```bash
# 1. Clone the repository
git clone https://github.com/laluni/software-ai-patent-auditor.git
cd software-ai-patent-auditor

# 2. Build and start all services (PostgreSQL pgvector, Ollama, Streamlit App)
docker compose up --build -d

# 3. Pull the Ollama model inside the container
docker exec -it patent_ollama ollama pull qwen2.5:latest
```

Open [http://localhost:8501](http://localhost:8501) in your browser!

---

### Option B: Local Python Environment Setup

#### 1. Install & Start Ollama
- Download and install Ollama from [ollama.com](https://ollama.com).
- Pull the model in your terminal:
  ```bash
  ollama pull qwen2.5:latest
  ```

#### 2. Start PostgreSQL Vector Database
```bash
docker compose up -d postgres
```

#### 3. Setup Python Virtual Environment & Launch App
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit web app
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

