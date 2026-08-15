import pytest
from src.rag import apply_semantic_floor, PatentRiskAnalysis, PatentAuditReport

def test_semantic_floor_upgrade():
    """
    Test that a LOW risk analysis is upgraded to MEDIUM if dense cosine similarity >= 0.50.
    """
    spec = "We use a token bucket algorithm to rate limit incoming API calls."
    candidates = [
        {
            "patent_number": "US-11983452-B2",
            "patent_title": "Token bucket rate-limited database index synchronization framework",
            "claim_text": "A system configured for rate limiting requests using token bucket leaky mechanisms."
        }
    ]
    
    initial_analysis = PatentRiskAnalysis(
        patent_id="US-11983452-B2",
        patent_title="Token bucket rate-limited database index synchronization framework",
        risk_level="LOW",
        overlapping_concepts=["token bucket"],
        legalese_translation="Rate limiting via tokens",
        suggested_design_around="Change algorithm"
    )
    
    report = PatentAuditReport(
        overall_risk="LOW",
        summary="Initial screening",
        analyses=[initial_analysis]
    )
    
    # Run the deterministic semantic floor
    updated_report = apply_semantic_floor(spec, candidates, report)
    
    # Because 'token bucket' text has high cosine similarity with the spec, it must be upgraded to MEDIUM
    assert updated_report.analyses[0].risk_level in ["MEDIUM", "HIGH"]
    assert updated_report.overall_risk in ["MEDIUM", "HIGH"]

def test_semantic_floor_no_upgrade_for_unrelated():
    """
    Test that an unrelated patent remains LOW risk when similarity is low.
    """
    spec = "We use a token bucket algorithm to rate limit incoming API calls."
    candidates = [
        {
            "patent_number": "US-9999999-B2",
            "patent_title": "Unrelated agricultural grain harvesting mechanism",
            "claim_text": "An agricultural apparatus comprising rotating blades for wheat harvesting."
        }
    ]
    
    initial_analysis = PatentRiskAnalysis(
        patent_id="US-9999999-B2",
        patent_title="Unrelated agricultural grain harvesting mechanism",
        risk_level="LOW",
        overlapping_concepts=[],
        legalese_translation="Grain harvesting machinery",
        suggested_design_around="None needed"
    )
    
    report = PatentAuditReport(
        overall_risk="LOW",
        summary="Initial screening",
        analyses=[initial_analysis]
    )
    
    updated_report = apply_semantic_floor(spec, candidates, report)
    assert updated_report.analyses[0].risk_level == "LOW"
    assert updated_report.overall_risk == "LOW"
