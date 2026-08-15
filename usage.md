# Usage & Step-by-Step Guide

This guide walks you through using the **Software & AI Patent Infringement Auditor** web interface.

---

## 1. Launching the Web Interface

Open your terminal in the project directory and run:

```bash
streamlit run app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 2. Running an Audit

### Option A: Using the Sample Specification
1. Click the **"💡 Load Sample Spec"** button above the input text area.
2. The input box will populate with a sample software architecture specification describing distributed GPU parameter optimization and rate-limited vector indexing.
3. Click **"🚀 Run Patent Audit (Pass 1 Screening)"**.

### Option B: Using a Custom Specification
1. Paste your proprietary technical design or architecture document into the text area.
2. Click **"🚀 Run Patent Audit (Pass 1 Screening)"**.

---

## 3. Interpreting the Audit Results

### Executive Summary & Risk Badges
- **HIGH RISK**: Direct structural or conceptual overlap with independent claims of published USPTO patents.
- **MEDIUM RISK**: Conceptual mechanism overlap (or cosine similarity $\ge 0.55$) requiring architectural design adjustments.
- **LOW RISK**: Distinct technical logic with minimal prior-art conflict.

### Detailed Claim Breakdown (Pass 1)
Expand each claim card to inspect:
- **Overlapping Concepts**: Extracted technical categories shared between your specification and the patent.
- **Plain-English Translation**: Simplified explanation of abstract patent legal phrasing.
- **Technical Design-Around Advice**: Concrete engineering recommendations on how to alter algorithms, component naming, or data flow to avoid conflict.

### On-Demand Deep Claim Audit (Pass 2)
1. Switch to the **On-Demand Deep Audit (Pass 2)** tab.
2. Select any candidate patent from the dropdown list.
3. Click **Run Deep Audit for Selected Patent**.
4. Review the clause-by-clause legal mapping table comparing independent claim elements against design specification features. Risk badges automatically synchronize across both passes.

---

## 4. Important Notice
This tool is designed to assist software engineers during architecture planning. It does not replace official legal counsel or formal freedom-to-operate clearance.
