import os
import json
import time
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUERY_LOG_FILE = os.path.join(LOGS_DIR, "query_monitoring_logs.json")
FEEDBACK_LOG_FILE = os.path.join(LOGS_DIR, "user_feedback_logs.json")

def _ensure_log_files():
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(QUERY_LOG_FILE):
        with open(QUERY_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    if not os.path.exists(FEEDBACK_LOG_FILE):
        with open(FEEDBACK_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

def log_query_event(
    query_text: str,
    top_rrf_score: float,
    db_latency_sec: float,
    pass1_latency_sec: float,
    safety_floor_triggered: bool,
    overall_risk: str,
    pass2_latency_sec: Optional[float] = None,
    guardrail_status: str = "IN_DOMAIN"
) -> Dict[str, Any]:
    """
    Logs an audit query transaction to the persistent monitoring log.
    """
    _ensure_log_files()
    
    event = {
        "id": f"qry_{int(time.time() * 1000)}",
        "timestamp": datetime.utcnow().isoformat(),
        "query_snippet": (query_text[:120] + "...") if len(query_text) > 120 else query_text,
        "query_length_chars": len(query_text),
        "top_rrf_score": round(float(top_rrf_score), 5),
        "db_latency_sec": round(float(db_latency_sec), 3),
        "pass1_latency_sec": round(float(pass1_latency_sec), 3),
        "pass2_latency_sec": round(float(pass2_latency_sec), 3) if pass2_latency_sec is not None else None,
        "total_latency_sec": round(float(db_latency_sec + pass1_latency_sec + (pass2_latency_sec or 0.0)), 3),
        "safety_floor_triggered": bool(safety_floor_triggered),
        "overall_risk": str(overall_risk).upper(),
        "guardrail_status": str(guardrail_status)
    }
    
    try:
        with open(QUERY_LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        logs = []
        
    logs.append(event)
    
    with open(QUERY_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
        
    return event

def log_feedback_event(
    patent_id: str,
    rating: int,  # +1 for Helpful, -1 for Inaccurate
    feedback_type: str = "General",
    user_comment: str = ""
) -> Dict[str, Any]:
    """
    Logs a human user rating and comment on a specific patent translation/audit.
    """
    _ensure_log_files()
    
    event = {
        "id": f"fb_{int(time.time() * 1000)}",
        "timestamp": datetime.utcnow().isoformat(),
        "patent_id": str(patent_id),
        "rating": int(rating),
        "rating_label": "Helpful" if rating > 0 else "Needs Improvement",
        "feedback_type": str(feedback_type),
        "user_comment": str(user_comment).strip()
    }
    
    try:
        with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        logs = []
        
    logs.append(event)
    
    with open(FEEDBACK_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
        
    return event

def get_query_logs_df() -> pd.DataFrame:
    """
    Returns query monitoring logs as a structured pandas DataFrame.
    """
    _ensure_log_files()
    try:
        with open(QUERY_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            # Return empty DataFrame with defined schema
            return pd.DataFrame(columns=[
                "id", "timestamp", "query_snippet", "query_length_chars",
                "top_rrf_score", "db_latency_sec", "pass1_latency_sec",
                "pass2_latency_sec", "total_latency_sec", "safety_floor_triggered",
                "overall_risk", "guardrail_status"
            ])
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        print(f"[Monitoring Error] Could not load query logs: {e}")
        return pd.DataFrame()

def get_feedback_logs_df() -> pd.DataFrame:
    """
    Returns user feedback logs as a structured pandas DataFrame.
    """
    _ensure_log_files()
    try:
        with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame(columns=[
                "id", "timestamp", "patent_id", "rating",
                "rating_label", "feedback_type", "user_comment"
            ])
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        print(f"[Monitoring Error] Could not load feedback logs: {e}")
        return pd.DataFrame()

def get_monitoring_summary_metrics() -> Dict[str, Any]:
    """
    Computes key aggregated metrics for the monitoring dashboard.
    """
    query_df = get_query_logs_df()
    feedback_df = get_feedback_logs_df()
    
    total_queries = len(query_df)
    if total_queries == 0:
        return {
            "total_queries": 0,
            "avg_latency_sec": 0.0,
            "safety_floor_rate_pct": 0.0,
            "avg_rrf_score": 0.0,
            "positive_feedback_pct": 100.0,
            "total_feedback": 0
        }
        
    avg_latency = query_df["total_latency_sec"].mean()
    floor_rate = (query_df["safety_floor_triggered"].sum() / total_queries) * 100.0
    avg_rrf = query_df["top_rrf_score"].mean()
    
    total_feedback = len(feedback_df)
    if total_feedback > 0:
        positive_count = (feedback_df["rating"] > 0).sum()
        pos_pct = (positive_count / total_feedback) * 100.0
    else:
        pos_pct = 100.0
        
    return {
        "total_queries": total_queries,
        "avg_latency_sec": round(float(avg_latency), 2),
        "safety_floor_rate_pct": round(float(floor_rate), 1),
        "avg_rrf_score": round(float(avg_rrf), 5),
        "positive_feedback_pct": round(float(pos_pct), 1),
        "total_feedback": total_feedback
    }
