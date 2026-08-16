import streamlit as st
import json
import time
import pandas as pd
from src.search import SearchEngine
from src.rag import (
    extract_keywords_from_design_doc,
    audit_patent_infringement,
    audit_patent_claims_pass2,
    verify_query_guardrail,
    sync_pass2_risk_to_report
)
from src.monitoring import (
    log_query_event,
    log_feedback_event,
    get_query_logs_df,
    get_feedback_logs_df,
    get_monitoring_summary_metrics
)

# Page Configuration
st.set_page_config(
    page_title="Software & AI Patent Infringement Auditor",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Software & AI Patent Infringement Auditor")
st.markdown("""
*Cross-reference software architecture specifications against **USPTO Patent Claims** using dense hybrid vector search, Reciprocal Rank Fusion (RRF), heuristic safety net, and local LLM reasoning.*
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
if "last_query_event" not in st.session_state:
    st.session_state.last_query_event = None

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_choice = st.selectbox("Ollama Model", ["qwen2.5:latest", "qwen2.5:3b", "llama3.1:8b"], index=0)
    top_k = st.slider("Top Claims to Retrieve", min_value=3, max_value=10, value=5)
    
    st.divider()
    st.markdown("### 📊 Quick System Health")
    metrics = get_monitoring_summary_metrics()
    st.metric("Total Queries Audited", metrics["total_queries"])
    st.metric("Avg Total Latency", f"{metrics['avg_latency_sec']}s")
    st.metric("User Approval Rating", f"{metrics['positive_feedback_pct']}%")

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
            start_total = time.time()
            st.session_state.pass2_report = None  # Reset Pass 2 on new search
            st.session_state.guardrail_warning = None
            
            # Step 1: Extract Search Keywords
            keywords = extract_keywords_from_design_doc(design_doc, model_name=model_choice)
            st.info(f"🔍 **Extracted USPTO Search Keywords:** `{', '.join(keywords)}`")

            # Step 2: Hybrid Retrieval
            t_db_start = time.time()
            engine = SearchEngine()
            retrieved = engine.search_prior_art(
                design_doc=design_doc,
                query_keywords=keywords,
                top_k=top_k
            )
            db_latency = time.time() - t_db_start
            st.session_state.retrieved_claims = retrieved

            # Step 3: Domain Guardrail Verification & Retrieval Confidence
            guard_check = verify_query_guardrail(design_doc, retrieved)
            st.session_state.guardrail_check = guard_check
            
            top_rrf = retrieved[0].get("rrf_score", 0.0) if retrieved else 0.0
            
            if not guard_check["is_in_domain"]:
                st.session_state.guardrail_warning = guard_check["reason"]
                st.session_state.audit_report = None
                # Log blocked query
                log_query_event(
                    query_text=design_doc,
                    top_rrf_score=top_rrf,
                    db_latency_sec=db_latency,
                    pass1_latency_sec=0.0,
                    safety_floor_triggered=False,
                    overall_risk="BLOCKED",
                    guardrail_status="BLOCKED"
                )
            else:
                # Step 4: LLM Infringement Audit (Pass 1)
                t_pass1_start = time.time()
                report = audit_patent_infringement(
                    design_doc=design_doc,
                    retrieved_claims=retrieved,
                    model_name=model_choice
                )
                pass1_latency = time.time() - t_pass1_start
                st.session_state.audit_report = report
                
                # Check if safety floor was triggered
                floor_triggered = any(
                    analysis.risk_level in ["MEDIUM", "HIGH"]
                    for analysis in report.analyses
                )
                
                # Log successful query
                q_event = log_query_event(
                    query_text=design_doc,
                    top_rrf_score=top_rrf,
                    db_latency_sec=db_latency,
                    pass1_latency_sec=pass1_latency,
                    safety_floor_triggered=floor_triggered,
                    overall_risk=report.overall_risk,
                    guardrail_status="IN_DOMAIN"
                )
                st.session_state.last_query_event = q_event
            
            elapsed = round(time.time() - start_total, 2)
            st.success(f"Pass 1 Candidate Screening completed in {elapsed} seconds!")

if st.session_state.guardrail_warning:
    st.divider()
    st.warning(f"🛡️ **Domain Guardrail Triggered:** {st.session_state.guardrail_warning}")
    st.info("💡 **Scope Reminder:** This auditor is specialized strictly for **Software, Artificial Intelligence, and Cloud Infrastructure Patents**. Please enter queries related to system architectures, ML algorithms, vector search, or database pipelines.")


# Display Audit Results & Monitoring Dashboard
st.divider()
st.subheader("2. Infringement Audit Report & Analytics")

# Four main tabs: Pass 1, Pass 2, Raw Data, and System Monitoring
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Claim Analyses (Pass 1)",
    "🔍 On-Demand Deep Audit (Pass 2)",
    "📄 Raw Retrieved Patents",
    "📊 System Monitoring & Analytics"
])

with tab1:
    if st.session_state.audit_report:
        report = st.session_state.audit_report
        
        # Single Overall Risk Badge
        risk = report.overall_risk.upper()
        if risk == "HIGH":
            st.error(f"⚠️ **OVERALL INFRINGEMENT RISK: {risk}**")
        elif risk == "MEDIUM":
            st.warning(f"⚡ **OVERALL INFRINGEMENT RISK: {risk}**")
        else:
            st.success(f"✅ **OVERALL INFRINGEMENT RISK: {risk}**")

        st.markdown(f"**Executive Summary:** {report.summary}")

        for idx, analysis in enumerate(report.analyses, 1):
            with st.expander(f"Claim #{idx}: Patent {analysis.patent_id} - {analysis.patent_title} (Risk: {analysis.risk_level})", expanded=(idx==1)):
                c1, c2 = st.columns([1, 2])
                c1.markdown(f"**Patent ID:** `{analysis.patent_id}`")
                c1.markdown(f"**Risk Level:** `{analysis.risk_level}`")
                c1.markdown(f"**Overlapping Concepts:** {', '.join(analysis.overlapping_concepts)}")
                
                c2.markdown(f"**📖 Plain-English Translation:**\n{analysis.legalese_translation}")
                c2.markdown(f"**💡 Technical Design-Around Advice:**\n{analysis.suggested_design_around}")
                
                st.markdown("---")
                st.markdown("**Rate this Claim Analysis & Advice:**")
                fb_c1, fb_c2, fb_c3 = st.columns([1, 1, 3])
                if fb_c1.button(f"👍 Helpful (#{idx})", key=f"up_{analysis.patent_id}_{idx}"):
                    log_feedback_event(
                        patent_id=analysis.patent_id,
                        rating=1,
                        feedback_type="Claim Translation",
                        user_comment="Helpful translation and advice"
                    )
                    st.toast(f"Thank you! Feedback recorded for {analysis.patent_id}.", icon="✅")
                if fb_c2.button(f"👎 Inaccurate (#{idx})", key=f"down_{analysis.patent_id}_{idx}"):
                    log_feedback_event(
                        patent_id=analysis.patent_id,
                        rating=-1,
                        feedback_type="Claim Translation",
                        user_comment="Inaccurate or low quality"
                    )
                    st.toast(f"Feedback noted for {analysis.patent_id}.", icon="⚠️")
    else:
        st.info("Enter a technical design specification above and click 'Run Patent Audit' to view Pass 1 claim results.")

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
                t_pass2_start = time.time()
                pass2_res = audit_patent_claims_pass2(
                    design_doc=design_doc,
                    target_patent=selected_patent,
                    model_name=model_choice
                )
                pass2_latency = time.time() - t_pass2_start
                st.session_state.pass2_report = pass2_res
                
                # Sync Pass 2 deep audit risk back to Pass 1 report & overall risk badge
                st.session_state.audit_report = sync_pass2_risk_to_report(
                    st.session_state.audit_report, pass2_res
                )
                
                # Update query log with Pass 2 latency
                if st.session_state.last_query_event:
                    st.session_state.last_query_event["pass2_latency_sec"] = round(pass2_latency, 3)
                    st.session_state.last_query_event["total_latency_sec"] += round(pass2_latency, 3)
                
                st.success(f"Pass 2 Deep Audit complete for {selected_patent.get('patent_number')} in {round(pass2_latency, 2)}s! Risk badges synchronized.")

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
    elif not st.session_state.retrieved_claims:
        st.info("Run Pass 1 screening first to populate candidate patents for deep analysis.")

with tab3:
    st.markdown("Raw USPTO patent claims retrieved via Hybrid Vector + Lexical RRF Search:")
    if st.session_state.retrieved_claims:
        st.json(st.session_state.retrieved_claims)
    else:
        st.info("No retrieved claims loaded yet.")

with tab4:
    st.markdown("### 📊 System Observability, Latency & Quality Monitoring")
    st.markdown("Real-time telemetry tracking query performance, retrieval distributions, heuristic safety nets, and human feedback.")

    q_df = get_query_logs_df()
    fb_df = get_feedback_logs_df()
    
    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    summary = get_monitoring_summary_metrics()
    kpi1.metric("Total Queries Audited", summary["total_queries"])
    kpi2.metric("Average Total Latency", f"{summary['avg_latency_sec']}s")
    kpi3.metric("Safety Floor Trigger Rate", f"{summary['safety_floor_rate_pct']}%")
    kpi4.metric("User Approval Rating", f"{summary['positive_feedback_pct']}%")
    
    st.divider()

    if not q_df.empty:
        # Chart Row 1: Latency Breakdown & Risk Verdicts
        c_row1_1, c_row1_2 = st.columns(2)
        
        with c_row1_1:
            st.markdown("#### 1. Latency Breakdown (DB Search vs. LLM Pass 1)")
            latency_chart_data = q_df[["db_latency_sec", "pass1_latency_sec"]].copy()
            latency_chart_data.columns = ["DB Search (sec)", "Pass 1 LLM (sec)"]
            st.bar_chart(latency_chart_data)

        with c_row1_2:
            st.markdown("#### 2. Risk Verdict Distribution")
            risk_counts = q_df["overall_risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            st.bar_chart(risk_counts.set_index("Risk Level"))

        st.divider()

        # Chart Row 2: Retrieval RRF Scores & Safety Net Activations
        c_row2_1, c_row2_2 = st.columns(2)

        with c_row2_1:
            st.markdown("#### 3. Top-1 Retrieval RRF Score Trend (Semantic Drift)")
            st.line_chart(q_df["top_rrf_score"].rename("Top-1 RRF Score"))

        with c_row2_2:
            st.markdown("#### 4. Cosine Safety Floor Trigger Rate")
            floor_counts = q_df["safety_floor_triggered"].map({True: "Upgraded by Floor", False: "Standard Assessment"}).value_counts().reset_index()
            floor_counts.columns = ["Status", "Count"]
            st.bar_chart(floor_counts.set_index("Status"))

        st.divider()

        # Chart Row 3: User Feedback Ratings
        st.markdown("#### 5. User Feedback Rating & Satisfaction")
        if not fb_df.empty:
            fb_counts = fb_df["rating_label"].value_counts().reset_index()
            fb_counts.columns = ["Rating", "Count"]
            st.bar_chart(fb_counts.set_index("Rating"))
        else:
            st.info("No user feedback submitted yet. Users can rate claim analyses in Tab 1 using 👍 or 👎.")

        st.divider()
        st.markdown("#### 📋 Live Transaction Audit Logs")
        st.dataframe(q_df.tail(20), use_container_width=True)
    else:
        st.info("No query logs recorded yet. Execute an audit above to view live system monitoring charts.")
