import os
import json
from src.search import SearchEngine
from src.db import PatentVectorStore
from src.ingest import ingest_patents_for_keywords

GROUND_TRUTH_FILE = os.path.join("data", "ground_truth.json")

def evaluate_retrieval_performance(top_k: int = 3):
    """
    Computes Hit Rate and MRR (Mean Reciprocal Rank) over ground-truth evaluation dataset.
    """
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"[Eval Error] Ground truth dataset not found at {GROUND_TRUTH_FILE}")
        return

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth_items = json.load(f)

    # Initialize store and seed dataset
    store = PatentVectorStore()
    keywords_list = ["neural", "vector", "privacy", "rate-limited", "federated", "image", "encryption", "message-queue", "audio", "dependency-injection", "hnsw", "anomaly"]
    ingest_patents_for_keywords(keywords_list, store, limit=20)
    engine = SearchEngine(vector_store=store)

    hits = 0
    reciprocal_ranks = []

    print(f"\n==========================================")
    print(f"   RUNNING RETRIEVAL EVALUATION (K={top_k})")
    print(f"==========================================\n")

    for idx, item in enumerate(ground_truth_items, 1):
        query = item["query"]
        expected_id = item["expected_patent_number"]
        keywords = item["keywords"]

        results = engine.search_prior_art(design_doc=query, query_keywords=keywords, top_k=top_k)
        retrieved_ids = [doc.get("patent_number") for doc in results]

        # Calculate Hit
        if expected_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(expected_id) + 1
            rr = 1.0 / rank
        else:
            rr = 0.0

        reciprocal_ranks.append(rr)

        print(f"Query #{idx}: '{query[:50]}...'")
        print(f"  Expected Patent: {expected_id}")
        print(f"  Retrieved IDs  : {retrieved_ids}")
        print(f"  Hit: {'[YES]' if rr > 0 else '[NO]'} (Rank: {1/rr if rr>0 else 'N/A'})\n")

    total = len(ground_truth_items)
    hit_rate = round(hits / total, 4)
    mrr = round(sum(reciprocal_ranks) / total, 4)

    print("==========================================")
    print("        FINAL EVALUATION METRICS          ")
    print("==========================================")
    print(f"  Total Test Queries : {total}")
    print(f"  Hit Rate @ {top_k}     : {hit_rate * 100}% ({hits}/{total})")
    print(f"  MRR (Mean Rec. Rank): {mrr}")
    print("==========================================\n")

    return {"hit_rate": hit_rate, "mrr": mrr}

if __name__ == "__main__":
    evaluate_retrieval_performance(top_k=3)
