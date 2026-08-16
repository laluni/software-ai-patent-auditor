"""
tests/test_monitoring_pillars.py

Pillar-by-pillar monitoring tests.
Each test targets an actual edge condition, not just "does the dashboard render."
"""
import time
import json
import os
import pytest
import pandas as pd

from src.monitoring import (
    log_query_event,
    log_feedback_event,
    update_pass2_latency,
    get_query_logs_df,
    get_feedback_logs_df,
    get_monitoring_summary_metrics,
    generate_request_id,
    check_rrf_drift,
    QUERY_LOG_FILE,
    FEEDBACK_LOG_FILE,
    RRF_DRIFT_THRESHOLD,
    RRF_DRIFT_WINDOW,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_logs(tmp_path, monkeypatch):
    """
    Redirect log files to a temporary directory for each test so tests
    are fully isolated and do not pollute the real data/ logs.
    """
    import src.monitoring as mon
    q_log = str(tmp_path / "query_monitoring_logs.json")
    f_log = str(tmp_path / "user_feedback_logs.json")
    monkeypatch.setattr(mon, "QUERY_LOG_FILE", q_log)
    monkeypatch.setattr(mon, "FEEDBACK_LOG_FILE", f_log)
    yield


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 1: RETRIEVAL QUALITY & SEMANTIC DRIFT
# ─────────────────────────────────────────────────────────────────────────────

class TestPillar1RetrievalDrift:

    def test_baseline_rrf_score_is_logged(self):
        """Baseline: high-quality query should log a score above drift threshold."""
        ev = log_query_event(
            query_text="distributed GPU transformer training",
            top_rrf_score=0.0325,
            db_latency_sec=0.018,
            pass1_latency_sec=3.5,
            safety_floor_triggered=False,
            overall_risk="HIGH",
        )
        assert ev["top_rrf_score"] == 0.0325
        assert ev["rrf_drift_detected"] is False
        assert ev["ingestion_triggered"] is False

    def test_single_low_rrf_does_not_trigger_ingestion(self):
        """
        Sustained drift vs. single dip:
        One bad query alone must NOT fire re-ingestion.
        """
        ev = log_query_event(
            query_text="ceramic mug glaze firing temperature",
            top_rrf_score=0.010,   # Very low but out-of-scope
            db_latency_sec=0.015,
            pass1_latency_sec=0.0,
            safety_floor_triggered=False,
            overall_risk="BLOCKED",
            guardrail_status="BLOCKED",
            guardrail_type="OUT_OF_SCOPE",
        )
        assert ev["ingestion_triggered"] is False, (
            "A single out-of-scope query must never trigger re-ingestion"
        )

    def test_out_of_scope_never_triggers_ingestion(self):
        """
        False-alarm check: even many consecutive out-of-scope (BLOCKED) queries
        must not trigger re-ingestion.
        """
        for _ in range(RRF_DRIFT_WINDOW + 2):
            ev = log_query_event(
                query_text="unrelated organic chemistry",
                top_rrf_score=0.005,
                db_latency_sec=0.010,
                pass1_latency_sec=0.0,
                safety_floor_triggered=False,
                overall_risk="BLOCKED",
                guardrail_status="BLOCKED",
                guardrail_type="OUT_OF_SCOPE",
            )
        assert ev["ingestion_triggered"] is False, (
            "Out-of-scope queries must never count toward drift window"
        )

    def test_sustained_in_domain_drift_triggers_ingestion(self):
        """
        Windowing: RRF_DRIFT_WINDOW consecutive in-domain low-RRF queries
        must trigger an ingestion alert.
        """
        events = []
        for i in range(RRF_DRIFT_WINDOW):
            ev = log_query_event(
                query_text=f"obscure vocab query {i}",
                top_rrf_score=RRF_DRIFT_THRESHOLD - 0.001,  # Just below threshold
                db_latency_sec=0.018,
                pass1_latency_sec=3.0,
                safety_floor_triggered=False,
                overall_risk="LOW",
                guardrail_status="IN_DOMAIN",
            )
            events.append(ev)

        # The last event in the window should fire
        last = events[-1]
        assert last["ingestion_triggered"] is True, (
            f"Expected ingestion trigger after {RRF_DRIFT_WINDOW} consecutive low-RRF in-domain queries"
        )
        assert last["ingestion_reason"] is not None

    def test_mixed_high_low_rrf_does_not_trigger(self):
        """
        Windowing: Alternating good and bad queries break the consecutive run
        and must NOT trigger re-ingestion.
        """
        for i in range(RRF_DRIFT_WINDOW * 2):
            score = 0.005 if i % 2 == 0 else 0.035   # alternating bad/good
            ev = log_query_event(
                query_text=f"query {i}",
                top_rrf_score=score,
                db_latency_sec=0.018,
                pass1_latency_sec=3.0,
                safety_floor_triggered=False,
                overall_risk="MEDIUM",
                guardrail_status="IN_DOMAIN",
            )
        assert ev["ingestion_triggered"] is False, (
            "Alternating good/bad queries must not trigger ingestion (no sustained drift)"
        )

    def test_request_ids_are_unique(self):
        """Each query gets a distinct request_id for cross-pillar traceability."""
        ids = {generate_request_id() for _ in range(20)}
        assert len(ids) == 20, "All request IDs must be unique"


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 2: GENERATION & SAFETY GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

class TestPillar2SafetyGuardrails:

    def test_similarity_floor_logged_as_distinct_type(self):
        """
        OUT_OF_SCOPE and SIMILARITY_FLOOR must be two distinct logged guardrail types,
        not conflated into a single bucket.
        """
        log_query_event(
            query_text="token bucket rate limiting",
            top_rrf_score=0.032,
            db_latency_sec=0.018,
            pass1_latency_sec=3.2,
            safety_floor_triggered=True,
            overall_risk="MEDIUM",
            guardrail_status="IN_DOMAIN",
            guardrail_type="SIMILARITY_FLOOR",
        )
        log_query_event(
            query_text="ceramic mug",
            top_rrf_score=0.005,
            db_latency_sec=0.015,
            pass1_latency_sec=0.0,
            safety_floor_triggered=False,
            overall_risk="BLOCKED",
            guardrail_status="BLOCKED",
            guardrail_type="OUT_OF_SCOPE",
        )

        df = get_query_logs_df()
        types = df["guardrail_type"].tolist()
        assert "SIMILARITY_FLOOR" in types, "Expected SIMILARITY_FLOOR to be logged"
        assert "OUT_OF_SCOPE" in types, "Expected OUT_OF_SCOPE to be logged"
        # Confirm they are different entries
        assert df[df["guardrail_type"] == "SIMILARITY_FLOOR"].shape[0] == 1
        assert df[df["guardrail_type"] == "OUT_OF_SCOPE"].shape[0] == 1

    def test_floor_trigger_rate_computed_correctly(self):
        """Dashboard floor rate must match manual count."""
        log_query_event("q1", 0.032, 0.018, 3.2, safety_floor_triggered=True, overall_risk="MEDIUM", guardrail_type="SIMILARITY_FLOOR")
        log_query_event("q2", 0.031, 0.018, 3.1, safety_floor_triggered=False, overall_risk="LOW", guardrail_type="NONE")
        log_query_event("q3", 0.030, 0.017, 3.0, safety_floor_triggered=True, overall_risk="HIGH", guardrail_type="SIMILARITY_FLOOR")

        summary = get_monitoring_summary_metrics()
        # 2 of 3 queries triggered floor → ~66.7%
        assert abs(summary["safety_floor_rate_pct"] - 66.7) < 0.2, (
            f"Expected ~66.7% floor trigger rate, got {summary['safety_floor_rate_pct']}"
        )

    def test_blocked_queries_counted_separately(self):
        """Out-of-scope guardrail blocks must be counted distinctly in summary metrics."""
        log_query_event("legit spec", 0.032, 0.018, 3.0, False, "LOW", guardrail_status="IN_DOMAIN")
        log_query_event("ceramic mug", 0.005, 0.015, 0.0, False, "BLOCKED",
                       guardrail_status="BLOCKED", guardrail_type="OUT_OF_SCOPE")

        summary = get_monitoring_summary_metrics()
        assert summary["guardrail_blocked_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 3: SYSTEM LATENCY & COMPUTE
# ─────────────────────────────────────────────────────────────────────────────

class TestPillar3Latency:

    def test_per_stage_latency_stored_separately(self):
        """
        Logs must store three distinct durations (DB, Pass 1, Pass 2),
        not just one wall-clock total.
        """
        req_id = generate_request_id()
        ev = log_query_event(
            query_text="HNSW vector indexing",
            top_rrf_score=0.033,
            db_latency_sec=0.018,
            pass1_latency_sec=3.8,
            safety_floor_triggered=False,
            overall_risk="HIGH",
            request_id=req_id,
        )
        assert ev["db_latency_ms"] == pytest.approx(18.0, abs=0.5)
        assert ev["pass1_latency_ms"] == pytest.approx(3800.0, abs=10.0)
        assert ev["pass2_latency_ms"] is None   # Not yet run

    def test_pass2_latency_updated_separately(self):
        """
        Pass 2 latency must be loggable separately after Pass 1,
        since it is user-triggered and optional.
        """
        req_id = generate_request_id()
        log_query_event(
            query_text="token bucket rate limiter",
            top_rrf_score=0.032,
            db_latency_sec=0.018,
            pass1_latency_sec=3.5,
            safety_floor_triggered=False,
            overall_risk="MEDIUM",
            request_id=req_id,
        )
        update_pass2_latency(req_id, pass2_latency_sec=8.2)

        df = get_query_logs_df()
        row = df[df["request_id"] == req_id].iloc[0]
        assert row["pass2_latency_sec"] == pytest.approx(8.2, abs=0.01)
        assert row["total_latency_sec"] == pytest.approx(0.018 + 3.5 + 8.2, abs=0.05)

    def test_total_is_sum_of_stages(self):
        """Total latency must equal sum of all stages that ran."""
        ev = log_query_event(
            query_text="PII redaction pipeline",
            top_rrf_score=0.031,
            db_latency_sec=0.020,
            pass1_latency_sec=4.1,
            safety_floor_triggered=False,
            overall_risk="LOW",
            pass2_latency_sec=7.5,
        )
        expected_total = round(0.020 + 4.1 + 7.5, 3)
        assert ev["total_latency_sec"] == pytest.approx(expected_total, abs=0.01)

    def test_pass2_not_included_in_pass1_metric(self):
        """
        Average latency metric must reflect ONLY queries where Pass 2 was NOT run,
        so Pass 1 screening speed is not artificially inflated by Pass 2 opt-in cost.
        """
        req1 = generate_request_id()
        req2 = generate_request_id()
        log_query_event("q1", 0.032, 0.018, 3.0, False, "LOW", request_id=req1)
        log_query_event("q2", 0.032, 0.018, 3.5, False, "LOW", request_id=req2)
        update_pass2_latency(req2, pass2_latency_sec=9.0)

        df = get_query_logs_df()
        pass1_only = df[df["pass2_latency_sec"].isna()]["pass1_latency_sec"]
        assert len(pass1_only) == 1
        assert pass1_only.iloc[0] == pytest.approx(3.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 4: HUMAN-IN-THE-LOOP FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────

class TestPillar4HumanFeedback:

    def test_feedback_attributed_to_specific_patent_and_claim(self):
        """
        Clicking 👍 on Claim #2 and 👎 on Claim #3 must be two separate,
        individually attributed log entries — not a single blob for the whole report.
        """
        req_id = generate_request_id()
        ev1 = log_feedback_event("US-11983452-B2", rating=1, request_id=req_id, claim_index=2,
                                  feedback_type="Claim Translation",
                                  user_comment="Good rate-limiting explanation")
        ev2 = log_feedback_event("US-11765432-B2", rating=-1, request_id=req_id, claim_index=3,
                                  feedback_type="Claim Translation",
                                  user_comment="Missed the HNSW equivalence — this IS overlapping")

        df = get_feedback_logs_df()
        assert len(df) == 2
        assert df.iloc[0]["claim_index"] == 2
        assert df.iloc[0]["rating"] == 1
        assert df.iloc[1]["claim_index"] == 3
        assert df.iloc[1]["rating"] == -1

    def test_user_comment_text_is_stored(self):
        """Comment text — the real bug-tracking signal — must be persisted, not discarded."""
        log_feedback_event("US-11983452-B2", rating=-1,
                           user_comment="Token bucket confusion: this IS a functional equivalent",
                           claim_index=1)
        df = get_feedback_logs_df()
        assert "Token bucket confusion" in df.iloc[0]["user_comment"]

    def test_feedback_shares_request_id_with_query(self):
        """
        Cross-pillar integration: feedback request_id must match the originating
        query request_id so events can be correlated across pillars.
        """
        req_id = generate_request_id()
        log_query_event("token bucket test", 0.032, 0.018, 3.5, True, "MEDIUM", request_id=req_id)
        log_feedback_event("US-11983452-B2", rating=-1, request_id=req_id, claim_index=1)

        q_df = get_query_logs_df()
        fb_df = get_feedback_logs_df()

        q_ids = set(q_df["request_id"].tolist())
        fb_ids = set(fb_df["request_id"].dropna().tolist())
        assert req_id in q_ids
        assert req_id in fb_ids
        assert q_ids & fb_ids == {req_id}, (
            "request_id must appear in BOTH query and feedback logs for cross-pillar correlation"
        )

    def test_zero_feedback_shows_none_not_100_percent(self):
        """
        Fresh session with no ratings must report None (rendered as N/A in UI),
        NOT a misleading 100% satisfaction from zero data.
        """
        # Add a query but no feedback
        log_query_event("some query", 0.032, 0.018, 3.5, False, "LOW")

        summary = get_monitoring_summary_metrics()
        assert summary["positive_feedback_pct"] is None, (
            "Zero feedback must return None, not 100.0 — that number is meaningless and misleading"
        )
        assert summary["total_feedback"] == 0

    def test_aggregate_satisfaction_matches_manual_count(self):
        """
        The displayed approval % must exactly match a manual recount from the raw log.
        Don't trust the metric blindly.
        """
        log_feedback_event("P1", 1, claim_index=1)
        log_feedback_event("P2", 1, claim_index=2)
        log_feedback_event("P3", -1, claim_index=1)
        log_feedback_event("P4", 1, claim_index=3)
        log_feedback_event("P5", -1, claim_index=2)

        summary = get_monitoring_summary_metrics()
        # 3 positive out of 5 → 60.0%
        assert summary["positive_feedback_pct"] == pytest.approx(60.0, abs=0.1), (
            f"Expected 60.0% (3/5 positive), got {summary['positive_feedback_pct']}"
        )
        assert summary["total_feedback"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-PILLAR INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossPillarIntegration:

    def test_all_four_pillars_share_request_id(self):
        """
        One request that triggers all four pillars:
        - Borderline similarity (Pillar 2)
        - Corpus drift signal (Pillar 1)
        - Timed end-to-end (Pillar 3)
        - User rates 👎 (Pillar 4)
        All four log entries must share the SAME request_id.
        """
        req_id = generate_request_id()

        # Pillars 1, 2, 3: query event
        ev = log_query_event(
            query_text="gradient descent inference weight tuning",
            top_rrf_score=0.018,          # borderline — near drift threshold
            db_latency_sec=0.019,
            pass1_latency_sec=3.9,
            safety_floor_triggered=True,  # Pillar 2: borderline similarity floor
            overall_risk="MEDIUM",
            guardrail_status="IN_DOMAIN",
            guardrail_type="SIMILARITY_FLOOR",
            request_id=req_id,
        )

        # Pillar 3: Pass 2 latency logged separately
        update_pass2_latency(req_id, pass2_latency_sec=8.5)

        # Pillar 4: user rates the result 👎
        fb_ev = log_feedback_event(
            patent_id="US-11842210-B2",
            rating=-1,
            feedback_type="Claim Translation",
            user_comment="Functional equivalence missed — gradient descent IS model optimization",
            request_id=req_id,
            claim_index=1,
        )

        # Verify cross-pillar correlation
        q_df = get_query_logs_df()
        fb_df = get_feedback_logs_df()

        matched_query = q_df[q_df["request_id"] == req_id]
        matched_feedback = fb_df[fb_df["request_id"] == req_id]

        assert len(matched_query) == 1,   "Query event must be found by request_id"
        assert len(matched_feedback) == 1, "Feedback event must be found by the same request_id"

        # Pillar 3: per-stage breakdown present
        row = matched_query.iloc[0]
        assert row["db_latency_ms"] > 0
        assert row["pass1_latency_ms"] > 0
        assert row["pass2_latency_sec"] == pytest.approx(8.5, abs=0.01)

        # Pillar 2: guardrail type logged correctly
        assert row["guardrail_type"] == "SIMILARITY_FLOOR"
        assert row["safety_floor_triggered"] == True  # noqa: E712 — must use == for numpy bools from pandas

        # Pillar 4: comment preserved, claim attributed
        fb_row = matched_feedback.iloc[0]
        assert "gradient descent" in fb_row["user_comment"]
        assert fb_row["claim_index"] == 1
