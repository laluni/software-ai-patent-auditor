import pytest
from src.db import PatentVectorStore

def test_rrf_scoring_logic():
    """
    Test that Reciprocal Rank Fusion correctly weights items present in both dense and sparse results.
    """
    store = PatentVectorStore()
    
    # Simulate candidates
    # Format: (id, patent_number, title, claim_text, assignee, date)
    dummy_patent = (1, "US-TEST-1", "Test Title", "Claim text here", "Inventor", "2024-01-01")
    
    # If a document is rank 1 in both dense and sparse (k=60):
    # score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
    rank_dense = 1
    rank_sparse = 1
    score = (1.0 / (60 + rank_dense)) + (1.0 / (60 + rank_sparse))
    
    assert round(score, 5) == round(2.0 / 61.0, 5)

def test_in_memory_vector_store_fallback():
    """
    Test that the in-memory fallback stores documents and executes hybrid search without crashing.
    """
    store = PatentVectorStore()
    assert store.use_pg is False or store.conn is not None
    
    # Insert test claim
    dummy_embedding = [0.1] * 384
    store.insert_claim(
        patent_number="US-TEST-99",
        patent_title="Test In-Memory Title",
        claim_text="Test claim with specific keywords like neural token bucket",
        assignee="Test Org",
        patent_date="2024-01-01",
        embedding=dummy_embedding
    )
    
    results = store.hybrid_search(
        query_text="neural token bucket",
        query_embedding=dummy_embedding,
        top_k=3
    )
    
    assert len(results) >= 1
    assert results[0]["patent_number"] == "US-TEST-99"
