import os
import pytest
import pandas as pd
from src.monitoring import (
    log_query_event,
    log_feedback_event,
    get_query_logs_df,
    get_feedback_logs_df,
    get_monitoring_summary_metrics
)

def test_query_event_logging():
    """
    Test logging a query transaction and validating stored schema.
    """
    event = log_query_event(
        query_text="Test design doc for distributed training",
        top_rrf_score=0.0325,
        db_latency_sec=0.045,
        pass1_latency_sec=3.21,
        safety_floor_triggered=True,
        overall_risk="HIGH",
        pass2_latency_sec=7.50,
        guardrail_status="IN_DOMAIN"
    )
    
    assert event["id"].startswith("qry_")
    assert event["overall_risk"] == "HIGH"
    assert event["safety_floor_triggered"] is True
    assert event["top_rrf_score"] == 0.0325
    assert round(event["total_latency_sec"], 3) == round(0.045 + 3.21 + 7.50, 3)

def test_feedback_event_logging():
    """
    Test logging human user feedback on a patent analysis.
    """
    event = log_feedback_event(
        patent_id="US-11983452-B2",
        rating=1,
        feedback_type="Claim Translation",
        user_comment="Accurate translation of rate limiting logic."
    )
    
    assert event["id"].startswith("fb_")
    assert event["patent_id"] == "US-11983452-B2"
    assert event["rating"] == 1
    assert event["rating_label"] == "Helpful"

def test_monitoring_dataframe_and_summary():
    """
    Test that log files are read cleanly into pandas DataFrames and metrics are aggregated.
    """
    q_df = get_query_logs_df()
    fb_df = get_feedback_logs_df()
    
    assert isinstance(q_df, pd.DataFrame)
    assert isinstance(fb_df, pd.DataFrame)
    assert len(q_df) >= 1
    assert len(fb_df) >= 1
    
    metrics = get_monitoring_summary_metrics()
    assert metrics["total_queries"] >= 1
    assert metrics["avg_latency_sec"] >= 0.0
    assert 0.0 <= metrics["positive_feedback_pct"] <= 100.0
