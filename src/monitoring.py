import os
import json
import time
import uuid
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUERY_LOG_FILE = os.path.join(LOGS_DIR, "query_monitoring_logs.json")
FEEDBACK_LOG_FILE = os.path.join(LOGS_DIR, "user_feedback_logs.json")

# --- Drift detection thresholds ---
RRF_DRIFT_THRESHOLD = 0.020       # Single-query floor below which we flag
RRF_DRIFT_WINDOW = 5              # Consecutive low-RRF queries before ingestion trigger fires
RRF_INGESTION_COOLDOWN_SECS = 300 # Don't re-trigger more than once per 5 min


def _ensure_log_files():
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(QUERY_LOG_FILE):
        with open(QUERY_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    if not os.path.exists(FEEDBACK_LOG_FILE):
        with open(FEEDBACK_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def generate_request_id() -> str:
    """
    Generates a short unique request ID that ties query and feedback events together.
    """
    return f"req_{uuid.uuid4().hex[:10]}"


def check_rrf_drift(current_rrf: float, guardrail_status: str) -> Dict[str, Any]:
    """
    Pillar 1: Windowed RRF drift detector.

    Checks whether recent queries show sustained retrieval degradation,
    distinguishing corpus drift (in-domain, low RRF) from out-of-scope
    queries (which should NOT trigger re-ingestion).

    Returns a dict with:
      - is_low_rrf: bool
      - consecutive_low_count: int
      - ingestion_triggered: bool
      - ingestion_reason: str | None
    """
    _ensure_log_files()

    # Out-of-scope queries must NOT trigger re-ingestion
    if guardrail_status == "BLOCKED":
        return {
            "is_low_rrf": current_rrf < RRF_DRIFT_THRESHOLD,
            "consecutive_low_count": 0,
            "ingestion_triggered": False,
            "ingestion_reason": "Out-of-scope query — drift detection skipped to avoid false alarm"
        }

    try:
        with open(QUERY_LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        logs = []

    # Consider only in-domain queries for drift window
    in_domain_logs = [
        l for l in logs
        if l.get("guardrail_status") == "IN_DOMAIN"
    ]

    # Count trailing consecutive low-RRF in-domain queries
    trailing = list(reversed(in_domain_logs[-RRF_DRIFT_WINDOW:]))
    consecutive_low = 0
    for entry in trailing:
        if entry.get("top_rrf_score", 1.0) < RRF_DRIFT_THRESHOLD:
            consecutive_low += 1
        else:
            break

    # Add current query to the window count
    is_low = current_rrf < RRF_DRIFT_THRESHOLD
    if is_low:
        consecutive_low += 1

    should_trigger = consecutive_low >= RRF_DRIFT_WINDOW

    # Cooldown: check when ingestion was last triggered
    if should_trigger:
        last_trigger_times = [
            l.get("ingestion_triggered_at")
            for l in logs
            if l.get("ingestion_triggered_at")
        ]
        if last_trigger_times:
            last_ts = max(last_trigger_times)
            elapsed = time.time() - last_ts
            if elapsed < RRF_INGESTION_COOLDOWN_SECS:
                should_trigger = False

    return {
        "is_low_rrf": is_low,
        "consecutive_low_count": consecutive_low,
        "ingestion_triggered": should_trigger,
        "ingestion_reason": (
            f"Sustained drift: {consecutive_low} consecutive in-domain queries below RRF threshold {RRF_DRIFT_THRESHOLD}"
            if should_trigger else None
        )
    }


def log_query_event(
    query_text: str,
    top_rrf_score: float,
    db_latency_sec: float,
    pass1_latency_sec: float,
    safety_floor_triggered: bool,
    overall_risk: str,
    pass2_latency_sec: Optional[float] = None,
    guardrail_status: str = "IN_DOMAIN",
    guardrail_type: str = "NONE",       # "NONE" | "SIMILARITY_FLOOR" | "OUT_OF_SCOPE"
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pillar 1 + 3: Logs an audit query transaction with per-stage latency breakdown,
    shared request_id for cross-pillar correlation, and drift detection.
    """
    _ensure_log_files()

    req_id = request_id or generate_request_id()

    drift = check_rrf_drift(top_rrf_score, guardrail_status)

    event = {
        "request_id": req_id,                                   # Cross-pillar correlation key
        "id": f"qry_{int(time.time() * 1000)}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_snippet": (query_text[:120] + "...") if len(query_text) > 120 else query_text,
        "query_length_chars": len(query_text),
        # --- Pillar 1: Retrieval Quality ---
        "top_rrf_score": round(float(top_rrf_score), 5),
        "rrf_drift_detected": drift["is_low_rrf"],
        "rrf_consecutive_low_count": drift["consecutive_low_count"],
        "ingestion_triggered": drift["ingestion_triggered"],
        "ingestion_triggered_at": time.time() if drift["ingestion_triggered"] else None,
        "ingestion_reason": drift["ingestion_reason"],
        # --- Pillar 2: Safety Guardrails ---
        "safety_floor_triggered": bool(safety_floor_triggered),
        "guardrail_status": str(guardrail_status),             # IN_DOMAIN | BLOCKED
        "guardrail_type": str(guardrail_type),                 # NONE | SIMILARITY_FLOOR | OUT_OF_SCOPE
        "overall_risk": str(overall_risk).upper(),
        # --- Pillar 3: Latency Breakdown ---
        "db_latency_ms": round(float(db_latency_sec) * 1000, 1),
        "pass1_latency_ms": round(float(pass1_latency_sec) * 1000, 1),
        "pass2_latency_ms": round(float(pass2_latency_sec) * 1000, 1) if pass2_latency_sec is not None else None,
        "total_latency_ms": round((db_latency_sec + pass1_latency_sec + (pass2_latency_sec or 0.0)) * 1000, 1),
        # Keep legacy sec fields for backward compat with existing charts
        "db_latency_sec": round(float(db_latency_sec), 3),
        "pass1_latency_sec": round(float(pass1_latency_sec), 3),
        "pass2_latency_sec": round(float(pass2_latency_sec), 3) if pass2_latency_sec is not None else None,
        "total_latency_sec": round((db_latency_sec + pass1_latency_sec + (pass2_latency_sec or 0.0)), 3),
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
    rating: int,              # +1 = Helpful, -1 = Needs Improvement
    feedback_type: str = "General",
    user_comment: str = "",
    request_id: Optional[str] = None,  # Pillar 4: ties back to the originating query
    claim_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Pillar 4: Logs per-claim human feedback attributed to the specific patent ID,
    claim index, and originating request_id for cross-pillar traceability.
    """
    _ensure_log_files()

    event = {
        "id": f"fb_{int(time.time() * 1000)}",
        "request_id": request_id,                # Cross-pillar correlation key
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patent_id": str(patent_id),
        "claim_index": claim_index,               # Which specific claim was rated
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


def update_pass2_latency(request_id: str, pass2_latency_sec: float):
    """
    Updates the Pass 2 latency on an existing query log entry by request_id.
    Pass 2 is user-triggered and optional — logged separately to avoid
    inflating Pass 1 screening cost measurements.
    """
    _ensure_log_files()
    try:
        with open(QUERY_LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
        for entry in reversed(logs):
            if entry.get("request_id") == request_id:
                entry["pass2_latency_sec"] = round(pass2_latency_sec, 3)
                entry["pass2_latency_ms"] = round(pass2_latency_sec * 1000, 1)
                entry["total_latency_sec"] = round(
                    entry.get("db_latency_sec", 0) +
                    entry.get("pass1_latency_sec", 0) +
                    pass2_latency_sec, 3
                )
                entry["total_latency_ms"] = round(entry["total_latency_sec"] * 1000, 1)
                break
        with open(QUERY_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"[Monitoring Error] Could not update pass2 latency: {e}")


def get_query_logs_df() -> pd.DataFrame:
    """Returns query monitoring logs as a structured pandas DataFrame."""
    _ensure_log_files()
    try:
        with open(QUERY_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame(columns=[
                "request_id", "id", "timestamp", "query_snippet", "query_length_chars",
                "top_rrf_score", "rrf_drift_detected", "rrf_consecutive_low_count",
                "ingestion_triggered", "safety_floor_triggered",
                "guardrail_status", "guardrail_type", "overall_risk",
                "db_latency_ms", "pass1_latency_ms", "pass2_latency_ms", "total_latency_ms",
                "db_latency_sec", "pass1_latency_sec", "pass2_latency_sec", "total_latency_sec",
            ])
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        return df
    except Exception as e:
        print(f"[Monitoring Error] Could not load query logs: {e}")
        return pd.DataFrame()


def get_feedback_logs_df() -> pd.DataFrame:
    """Returns user feedback logs as a structured pandas DataFrame."""
    _ensure_log_files()
    try:
        with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            return pd.DataFrame(columns=[
                "id", "request_id", "timestamp", "patent_id", "claim_index",
                "rating", "rating_label", "feedback_type", "user_comment"
            ])
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
        return df
    except Exception as e:
        print(f"[Monitoring Error] Could not load feedback logs: {e}")
        return pd.DataFrame()


def get_monitoring_summary_metrics() -> Dict[str, Any]:
    """Computes key aggregated metrics for the monitoring dashboard."""
    query_df = get_query_logs_df()
    feedback_df = get_feedback_logs_df()

    total_queries = len(query_df)
    total_feedback = len(feedback_df)

    # Compute feedback independently — it must not be gated on queries
    if total_feedback > 0:
        positive_count = (feedback_df["rating"] > 0).sum()
        pos_pct: Optional[float] = round(float(positive_count / total_feedback * 100.0), 1)
    else:
        pos_pct = None  # Explicitly None — rendered as "N/A" in UI, not 100%

    if total_queries == 0:
        return {
            "total_queries": 0,
            "avg_latency_sec": 0.0,
            "safety_floor_rate_pct": 0.0,
            "avg_rrf_score": 0.0,
            "positive_feedback_pct": pos_pct,
            "total_feedback": total_feedback,
            "ingestion_triggered_count": 0,
            "guardrail_blocked_count": 0,
        }

    avg_latency = query_df["total_latency_sec"].mean()
    floor_rate = (query_df["safety_floor_triggered"].sum() / total_queries) * 100.0
    avg_rrf = query_df["top_rrf_score"].mean()
    ingestion_count = int(query_df.get("ingestion_triggered", pd.Series([False])).sum()) if "ingestion_triggered" in query_df.columns else 0
    blocked_count = int((query_df["guardrail_status"] == "BLOCKED").sum()) if "guardrail_status" in query_df.columns else 0

    return {
        "total_queries": total_queries,
        "avg_latency_sec": round(float(avg_latency), 2),
        "safety_floor_rate_pct": round(float(floor_rate), 1),
        "avg_rrf_score": round(float(avg_rrf), 5),
        "positive_feedback_pct": pos_pct,
        "total_feedback": total_feedback,
        "ingestion_triggered_count": ingestion_count,
        "guardrail_blocked_count": blocked_count,
    }
