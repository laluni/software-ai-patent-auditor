import os
import dlt
from typing import Iterator, Dict, Any, List
from sentence_transformers import SentenceTransformer
from src.api_client import fetch_patents_from_uspto
from src.db import PatentVectorStore

_EMBED_MODEL = None

def get_embedding_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print("[DLT Pipeline] Loading embedding model BAAI/bge-small-en-v1.5...")
        _EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _EMBED_MODEL

def generate_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

@dlt.resource(name="uspto_patents", write_disposition="merge", primary_key="patent_number")
def uspto_patent_source(keywords: List[str], limit_per_keyword: int = 10) -> Iterator[Dict[str, Any]]:
    """
    DLT Generator Resource that yields normalized patent records from the USPTO API.
    """
    raw_patents = fetch_patents_from_uspto(keywords, limit=limit_per_keyword)
    print(f"[DLT Resource] Extracted {len(raw_patents)} raw patent records from USPTO source.")

    for patent in raw_patents:
        p_num = patent.get("patent_number", "US-UNKNOWN")
        p_title = patent.get("patent_title", "Untitled Patent")
        p_abstract = patent.get("patent_abstract", "")
        assignee = patent.get("assignee_organization") or "Independent Inventor"
        p_date = patent.get("patent_date", "2024-01-01")
        
        claim_text = f"Patent Title: {p_title}. Claim Text: {p_abstract}"
        
        yield {
            "patent_number": p_num,
            "patent_title": p_title,
            "patent_abstract": p_abstract,
            "claim_text": claim_text,
            "assignee_organization": assignee,
            "patent_date": p_date
        }

def run_dlt_pipeline(keywords: List[str], vector_store: PatentVectorStore = None, limit_per_keyword: int = 10):
    """
    Runs the DLT automated pipeline:
    1. Extracts and normalizes patent data via DLT resource.
    2. Ingests normalized records and generated dense embeddings into PostgreSQL + pgvector.
    """
    if vector_store is None:
        vector_store = PatentVectorStore()

    print(f"[DLT Ingestion Pipeline] Starting automated DLT pipeline for keywords: {keywords}")
    
    # 1. Run DLT extraction resource
    resource = uspto_patent_source(keywords=keywords, limit_per_keyword=limit_per_keyword)
    
    # 2. Populate PostgreSQL pgvector knowledge base
    count = 0
    for record in resource:
        embedding = generate_embedding(record["claim_text"])
        vector_store.insert_claim(
            patent_number=record["patent_number"],
            patent_title=record["patent_title"],
            claim_text=record["claim_text"],
            assignee=record["assignee_organization"],
            patent_date=record["patent_date"],
            embedding=embedding
        )
        count += 1

    print(f"[DLT Ingestion Pipeline] Successfully normalized and loaded {count} patent vectors into PostgreSQL!")
    return count

if __name__ == "__main__":
    run_dlt_pipeline(["neural", "vector", "database"], limit_per_keyword=5)
