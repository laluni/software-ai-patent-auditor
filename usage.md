# Usage & Step-by-Step Guide

This guide walks you through using the **Software & AI Patent Infringement Auditor** application.

---

## 🚀 1. Launching the Web Interface

Open your terminal in the `ai-patent-auditor` directory and run:

```bash
python -m streamlit run app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)**.

---

## 💡 2. Running an Audit

### Option A: Using the Sample Spec
1. Click the **"💡 Load Sample Spec"** button above the text area.
2. The input box will populate with a sample software architecture specification describing distributed GPU transformer optimization and rate-limited vector indexing.
3. Click **"🚀 Run Patent Audit (Pass 1 Screening)"**.

### Option B: Using Your Own Specification
1. Paste your technical design specification into the text area.
2. Click **"🚀 Run Patent Audit (Pass 1 Screening)"**.

---

## 📊 3. Interpreting Audit Results

### Executive Summary & Synchronized Risk Badge
- 🔴 **HIGH RISK**: Direct or structural overlap with independent claims of published USPTO patents.
- 🟡 **MEDIUM RISK**: Conceptual mechanism overlap (or cosine similarity $\ge 0.50$) requiring architectural design adjustments.
- 🟢 **LOW RISK**: Novel technical logic with minimal prior-art conflict.

### Detailed Claim Breakdown (Pass 1)
Expand each claim card to view:
- **Overlapping Concepts**: Extracted technical categories shared between your spec and the patent.
- **Plain-English Translation**: Demystified explanation of the obfuscated legal language.
- **Technical Design-Around Advice**: Specific recommendations on how to alter module naming, control flow, or algorithms to avoid patent infringement.

### On-Demand Deep Claim Audit (Pass 2)
1. Switch to the **🔍 On-Demand Deep Audit (Pass 2)** tab.
2. Select any candidate patent from the dropdown list.
3. Click **🔍 Run Deep Audit for Selected Patent**.
4. Review the element-by-element legal mapping table comparing independent claim clauses against design doc features. Risk badges across Pass 1 and Pass 2 automatically synchronize.

