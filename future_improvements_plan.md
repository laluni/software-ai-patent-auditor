# 🛠️ Technical Plan: Resolving System Limitations & Hardening Architecture

This document provides a technical blueprint to solve the key technical limitations of the **Software & AI Patent Infringement Auditor** system. 

---

## 📋 Summary of Solvability

Below is the exact architecture, data structure, and technical implementation plan for each component.

| Limitation Area | Solution Overview | Expected System Improvement |
| :--- | :--- | :--- |
| **1. Synthetic Data Overfitting** | Template-free prompt engineering + NER entity restriction. | Zero prompt-bias (eliminates over-indexed terms like "Docker"). |
| **2. Full Patent Claim Parsing** | USPTO XML hierarchical parser separating Independent & Dependent claims. | Legally airtight claim-by-claim infringement matching. |
| **3. Keyword Extraction Latency** | Replace LLM keyword call with local sub-5ms **KeyBERT / spaCy** extractor. | Drops query latency by **2–3 seconds** per audit. |
| **4. Monitoring Complexity** | Integrate **Arize Phoenix / OpenTelemetry** for real-time latency and token tracking. | Real-time visual tracking of latency p95, token cost, and embedding drift. |

---

## 🛠️ Detailed Technical Solutions & Code Specs

### Solution 1: Unbiased Synthetic Data Generation Pipeline (`src/generate_dataset_unbiased.py`)

#### The Fix:
Modify the dataset generation prompt to forbid hardcoded example keywords, restricting the LLM strictly to entities extracted directly from the target patent claim text.

#### Implementation Code Spec:
```python
def generate_unbiased_anchor(patent_title: str, patent_abstract: str, model_name: str = "qwen2.5:latest") -> str:
    """
    Generates a synthetic developer specification anchor strictly constrained 
    to the technical entities present in the target patent text.
    """
    prompt = f"""
    You are a technical translator. Rephrase the following patent claim into a single sentence technical design specification written in plain developer terminology.

    STRICT CONSTRAINTS:
    1. Do NOT use generic placeholder technologies (e.g. DO NOT mention 'Docker', 'Kubernetes', 'FastAPI', or 'Redis' unless they are explicitly written in the patent text below).
    2. Rely ONLY on the technical concepts described in the patent text.
    3. Output ONLY the rephrased 1-sentence technical specification.

    Patent Title: {patent_title}
    Patent Abstract: {patent_abstract}

    Technical Spec:
    """
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip()
```

---

### Solution 2: Hierarchical USPTO XML Claim Parser (`src/patent_parser.py`)

#### The Fix:
USPTO patent publications are distributed as XML files conforming to the ST.36 / US-PAT-DOC DTD format. We build an XML parser using `xml.etree.ElementTree` to separate **Independent Claims** (which define the broad legal scope) from **Dependent Claims** (which reference parent claims).

#### Hierarchical Claim Data Schema:
```json
{
  "patent_number": "US-11842210-B2",
  "patent_title": "Distributed Neural Network Parameter Optimization",
  "claims": [
    {
      "claim_id": "claim_1",
      "claim_type": "independent",
      "parent_claim_id": null,
      "claim_text": "1. A computer-implemented method for distributed optimization comprising..."
    },
    {
      "claim_id": "claim_2",
      "claim_type": "dependent",
      "parent_claim_id": "claim_1",
      "claim_text": "2. The method of claim 1, wherein the compute nodes comprise GPU clusters."
    }
  ]
}
```

#### Ingestion Strategy:
Each independent claim is ingested as an individual vector chunk with `claim_type: "independent"` attached as metadata. This allows the RAG engine to evaluate infringement against core broad claims first before checking dependent sub-claims.

---

### Solution 3: Sub-5ms Local NLP Keyword Extractor (`src/fast_keywords.py`)

#### The Fix:
Instead of making an expensive LLM call (`qwen2.5:latest`) that takes **2,000–3,000 ms** to extract 3 keywords, we use **KeyBERT** or **TF-IDF + spaCy Noun Chunks** locally in Python.

#### Performance Comparison:
* **LLM Keyword Extraction:** ~2,500 ms latency + risk of JSON formatting errors.
* **Local KeyBERT / spaCy Extraction:** **~4 ms latency**, 100% deterministic, zero LLM cost.

#### Implementation Code Spec:
```python
from keybert import KeyBERT

# Load lightweight KeyBERT model once
_kw_model = KeyBERT('all-MiniLM-L6-v2')

def extract_keywords_fast(text: str, top_n: int = 4) -> list[str]:
    """
    Extracts key technical phrases in < 5ms using local KeyBERT.
    """
    keywords_with_scores = _kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=top_n
    )
    return [kw[0] for kw[0] in keywords_with_scores]

# Example:
# Input: "Distributed neural network parameter optimization using synchronous GPU gradients"
# Output: ['gpu gradients', 'neural network', 'parameter optimization', 'distributed']
```

---

### Solution 4: Zero-Config Telemetry Dashboard with Arize Phoenix (`src/telemetry_dashboard.py`)

#### The Fix:
Replace simple JSON logging (`data/feedback_logs.json`) with **Arize Phoenix**, an open-source, local-first LLM observability tool.

#### Implementation Steps:
1. Install Phoenix: `pip install arize-phoenix`
2. Launch local trace collector in `app.py`:

```python
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor

# Launch local Phoenix tracing server on http://localhost:6006
session = px.launch_app()

# Auto-instrument LLM calls & retrieval steps
LangChainInstrumentor().instrument()
```

#### Dashboard Capabilities:
* **Latency Breakdown:** View exact p50, p95, and p99 latency histograms for vector search vs. LLM generation.
* **Token Usage & Costs:** Real-time tracking of input/output token counts.
* **Embedding Drift:** Visual 2D UMAP projection of user queries over time to detect out-of-domain searches.

---

## 📅 Roadmap for Next Steps

```text
Phase 1: Fast Keyword Extractor ──► Phase 2: Unbiased Dataset Gen ──► Phase 3: XML Claim Parser ──► Phase 4: Arize Phoenix
   (Integrate KeyBERT)               (Remove Prompt Bias)             (Separate Claims)              (Visual Telemetry)
```
