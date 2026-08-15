# System Architecture & Processing Workflow

This document details the technical architecture, data handling, search mechanisms, and multi-pass RAG workflow for the **Software & AI Patent Infringement Auditor**.

---

## 1. System Overview & Scope

The **Software & AI Patent Infringement Auditor** enables software engineers to cross-reference proprietary architecture specifications against **USPTO Patent Claims** to detect potential prior-art overlap and generate technical design-around recommendations.

### Domain Scope
The indexed repository focuses on **Software, Artificial Intelligence, and Cloud Infrastructure Patents**:
- **Software Engineering**: Dependency injection, microservice pipelines, PII redaction, API security.
- **Artificial Intelligence & ML**: Distributed GPU parameter optimization, federated learning, attention mechanisms, autoencoders.
- **Infrastructure & DBs**: Vector database indexing (HNSW), token bucket rate-limiters, message queues, homomorphic key rotation.

---

## 2. System Architecture Diagram

```mermaid
flowchart TD
    A[User Inputs Technical Spec] --> B[RRF Domain Guardrail Check]
    B -->|Out of Domain| C[Halt & Show Guardrail Warning]
    B -->|In Domain| D[Extract Keywords via qwen2.5 & Generate Dense Query Embedding]
    D --> E{Local DB Populated?}
    E -->|No| F[Fetch & Normalize Patents via dlt Pipeline]
    F --> G[Generate Embeddings & Save to PostgreSQL pgvector]
    E -->|Yes| H[Hybrid Search in PostgreSQL pgvector]
    G --> H
    H --> I[Dense Vector Cosine Top 20 + Sparse Lexical FTS Top 20]
    I --> J[Reciprocal Rank Fusion - RRF Score Calculation]
    J --> K[Select Top 3 to 5 Candidate Patents]
    K --> L[Pass 1: Structured Per-Patent LLM Screening - qwen2.5]
    L --> M[Apply Cosine Similarity Safety Floor sim >= 0.55]
    M --> N[Display Single Synchronized Overall Risk Badge: HIGH / MEDIUM / LOW]
    N -->|User Selects Single Patent for Deep Audit| O[Pass 2: On-Demand Deep Line-by-Line Claim Audit]
    O --> P[Element Mapping Table & Risk Badge Synchronization across Tabs]
```

---

## 3. Step-by-Step Processing Pipeline

### Step 1: Input & RRF Domain Guardrail Filter
- **Location**: `src/rag.py` (`verify_query_guardrail()`) & `app.py`
- **Process**:
  1. The user inputs their technical architecture specification into Streamlit.
  2. The guardrail evaluates the candidate Reciprocal Rank Fusion (RRF) scores (`top_score < 0.018`).
  3. If out-of-domain (e.g., mechanical or chemical queries), execution halts early to prevent hallucinations and save local compute.

---

### Step 2: Query Embedding & Keyword Extraction
- **Locations**: `src/dlt_ingest.py` (`generate_embedding()`) & `src/rag.py` (`extract_keywords_from_design_doc()`)
- **Process**:
  1. **Query Embedding**: The specification is encoded by `BAAI/bge-small-en-v1.5` into a 384-dimensional dense vector.
  2. **Keyword Extraction**: `qwen2.5:latest` extracts 3 to 4 core technical search terms (*e.g., `["vector database", "rate limiter", "neural network"]`*).

---

### Step 3: Automated Data Ingestion (`dlt`)
- **Location**: `src/dlt_ingest.py`
- **Process**:
  1. If the local vector store is unpopulated, `dlt` streams records from the USPTO PatentsView API.
  2. Primary key deduplication (`primary_key="patent_number"`, `write_disposition="merge"`) prevents duplicate vector rows.
  3. Inserts embeddings, metadata, and full-text search columns into PostgreSQL `pgvector`.

---

### Step 4: Hybrid Search & Reciprocal Rank Fusion (RRF)
- **Location**: `src/db.py` (`hybrid_search()`)
- **Process**:
  1. Executes dense cosine distance search over `embedding vector(384)` to retrieve Top-20 candidates.
  2. Executes sparse BM25 text search over `to_tsvector('english', claim_text)` to retrieve Top-20 candidates.
  3. Merges and scores candidates using Reciprocal Rank Fusion ($k=60$):
     $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
  4. Returns the Top-K candidate patents.

---

### Step 5: Pass 1 Screening & Cosine Safety Floor
- **Location**: `src/rag.py` (`audit_patent_infringement()`, `apply_semantic_floor()`)
- **Process**:
  1. Prompts `qwen2.5` using the Doctrine of Equivalents to evaluate functional equivalence.
  2. Computes programmatic cosine similarity between the input specification vector $\vec{u}$ and candidate claim vectors $\vec{v}$.
  3. If $\text{similarity} \ge 0.55$, automatically upgrades any `LOW` risk badge to at least `MEDIUM`.

---

### Step 6: Pass 2 On-Demand Deep Claim Audit
- **Location**: `src/rag.py` (`audit_single_patent_deep()`)
- **Process**:
  1. The user selects a specific candidate patent in the UI.
  2. The LLM breaks down the independent claim into individual elements and outputs a structured clause-by-clause comparison table.
  3. Updates and synchronizes risk badges across both UI views.
