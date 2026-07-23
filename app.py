import streamlit as st
import json
import time
from src.search import SearchEngine
from src.rag import (
    extract_keywords_from_design_doc,
    audit_patent_infringement,
    audit_patent_claims_pass2,
    verify_query_guardrail,
    sync_pass2_risk_to_report
)

# Page Configuration
st.set_page_config(
    page_title="Software & AI Patent Infringement Auditor",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Software & AI Patent Infringement Auditor")
st.markdown("""
*Cross-reference software architecture specifications against **USPTO Patent Claims** using dense hybrid vector search, cross-lingual translation, Doctrine of Equivalents semantic safety net, and Ollama LLM reasoning.*
""")

# Initialize Session State
if "audit_report" not in st.session_state:
    st.session_state.audit_report = None
if "pass2_report" not in st.session_state:
    st.session_state.pass2_report = None
if "retrieved_claims" not in st.session_state:
    st.session_state.retrieved_claims = None
if "design_doc_text" not in st.session_state:
    st.session_state.design_doc_text = ""
if "guardrail_check" not in st.session_state:
    st.session_state.guardrail_check = None
if "guardrail_warning" not in st.session_state:
    st.session_state.guardrail_warning = None

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox("Ollama Model", ["qwen2.5:latest", "qwen2.5:3b", "llama3.1:8b"], index=0)
    top_k = st.slider("Top Claims to Retrieve", min_value=3, max_value=10, value=5)

# Main Interface: Input Section
st.subheader("1. Enter Technical Design Specification")

sample_btn_col1, sample_btn_col2 = st.columns([1, 4])
if sample_btn_col1.button("💡 Load Sample Spec"):
    st.session_state.design_doc_text = """We are building a distributed neural network training system that optimizes transformer model weights using synchronous gradient updates across parallel GPU compute nodes. The system includes an asynchronous vector index synchronization pipeline with non-blocking token-bucket rate limiting."""

design_doc = st.text_area(
    "Paste your software architecture, module design doc, or system specification:",
    value=st.session_state.design_doc_text,
    height=180,
    placeholder="Describe your system architecture, ML algorithms, database indexing, or pipeline logic..."
)

if st.button("🚀 Run Patent Audit (Pass 1 Screening)", type="primary"):
    if not design_doc.strip():
        st.warning("Please enter a design specification before running the audit.")
    else:
        with st.spinner("Extracting technical search terms and querying USPTO database..."):
            start_time = time.time()
            st.session_state.pass2_report = None  # Reset Pass 2 on new search
            st.session_state.guardrail_warning = None
            
            # Step 1: Extract Search Keywords
            keywords = extract_keywords_from_design_doc(design_doc, model_name=model_choice)
            st.info(f"🔍 **Extracted USPTO Search Keywords:** `{', '.join(keywords)}`")

            # Step 2: Hybrid Retrieval
            engine = SearchEngine()
            retrieved = engine.search_prior_art(
                design_doc=design_doc,
                query_keywords=keywords,
                top_k=top_k
            )
            st.session_state.retrieved_claims = retrieved

            # Step 3: Domain Guardrail Verification & Retrieval Confidence
            guard_check = verify_query_guardrail(design_doc, retrieved)
            st.session_state.guardrail_check = guard_check
            
            if not guard_check["is_in_domain"]:
                st.session_state.guardrail_warning = guard_check["reason"]
                st.session_state.audit_report = None
            else:
                # Step 4: LLM Infringement Audit (Pass 1)
                report = audit_patent_infringement(
                    design_doc=design_doc,
                    retrieved_claims=retrieved,
                    model_name=model_choice
                )
                st.session_state.audit_report = report
            
            elapsed = round(time.time() - start_time, 2)
            st.success(f"Pass 1 Candidate Screening completed in {elapsed} seconds!")

if st.session_state.guardrail_warning:
    st.divider()
    st.warning(f"🛡️ **Domain Guardrail Triggered:** {st.session_state.guardrail_warning}")
    st.info("💡 **Scope Reminder:** This auditor is specialized strictly for **Software, Artificial Intelligence, and Cloud Infrastructure Patents**. Please enter queries related to system architectures, ML algorithms, vector search, or database pipelines.")


# Display Audit Results
if st.session_state.audit_report:
    report = st.session_state.audit_report

    st.divider()
    st.subheader("2. Infringement Audit Report")

    # Single Overall Risk Badge
    risk = report.overall_risk.upper()
    if risk == "HIGH":
        st.error(f"⚠️ **OVERALL INFRINGEMENT RISK: {risk}**")
    elif risk == "MEDIUM":
        st.warning(f"⚡ **OVERALL INFRINGEMENT RISK: {risk}**")
    else:
        st.success(f"✅ **OVERALL INFRINGEMENT RISK: {risk}**")

    st.markdown(f"**Executive Summary:** {report.summary}")

    # Tabs for detailed views
    tab1, tab2, tab3 = st.tabs(["📋 Claim Analyses (Pass 1)", "🔍 On-Demand Deep Audit (Pass 2)", "📄 Raw Retrieved Patents"])

    with tab1:
        for idx, analysis in enumerate(report.analyses, 1):
            with st.expander(f"Claim #{idx}: Patent {analysis.patent_id} - {analysis.patent_title} (Risk: {analysis.risk_level})", expanded=(idx==1)):
                c1, c2 = st.columns([1, 2])
                c1.markdown(f"**Patent ID:** `{analysis.patent_id}`")
                c1.markdown(f"**Risk Level:** `{analysis.risk_level}`")
                c1.markdown(f"**Overlapping Concepts:** {', '.join(analysis.overlapping_concepts)}")
                
                c2.markdown(f"**📖 Plain-English Translation:**\n{analysis.legalese_translation}")
                c2.markdown(f"**💡 Technical Design-Around Advice:**\n{analysis.suggested_design_around}")

    with tab2:
        st.markdown("### 🔬 On-Demand Pass 2: Deep Line-by-Line Claim Audit")
        st.markdown("Select a specific patent from the retrieved candidates to perform an element-by-element legal mapping of its Independent Claim.")

        if st.session_state.retrieved_claims:
            patent_options = {
                f"{p.get('patent_number', 'US-UNKNOWN')} - {p.get('patent_title', 'Untitled')}": p
                for p in st.session_state.retrieved_claims
            }
            selected_patent_label = st.selectbox("Select Patent to Deep-Audit:", list(patent_options.keys()))
            selected_patent = patent_options[selected_patent_label]

            if st.button("🔍 Run Deep Audit for Selected Patent", type="secondary"):
                with st.spinner(f"Analyzing Claim 1 of {selected_patent.get('patent_number')} line-by-line..."):
                    pass2_res = audit_patent_claims_pass2(
                        design_doc=design_doc,
                        target_patent=selected_patent,
                        model_name=model_choice
                    )
                    st.session_state.pass2_report = pass2_res
                    # Sync Pass 2 deep audit risk back to Pass 1 report & overall risk badge
                    st.session_state.audit_report = sync_pass2_risk_to_report(
                        st.session_state.audit_report, pass2_res
                    )
                    st.success(f"Pass 2 Deep Audit complete for {selected_patent.get('patent_number')}! Risk badges synchronized.")

        if st.session_state.pass2_report:
            p2 = st.session_state.pass2_report
            st.divider()
            st.markdown(f"**Pass 2 Legal Summary:** {p2.executive_summary}")
            
            for audit in p2.deep_audits:
                st.markdown(f"#### 📜 Patent {audit.patent_number}: {audit.patent_title}")
                m1, m2 = st.columns([1, 3])
                m1.metric("Infringement Probability", f"{audit.infringement_probability_pct}%")
                m1.markdown(f"**Risk Level:** `{audit.risk_level}`")

                m2.markdown("**Element-by-Element Infringement Mapping Table:**")
                table_data = []
                for elem in audit.element_matches:
                    table_data.append({
                        "Independent Claim Element": elem.claim_element,
                        "Matching Design Doc Feature": elem.matching_design_feature,
                        "Overlap Status": "⚠️ OVERLAP" if elem.is_overlapping else "✅ DISTINCT"
                    })
                m2.table(table_data)
                st.divider()

    with tab3:
        st.markdown("Raw USPTO patent claims retrieved via Hybrid Vector + Lexical RRF Search:")
        st.json(st.session_state.retrieved_claims)
