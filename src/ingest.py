import os
from sentence_transformers import SentenceTransformer
from src.api_client import fetch_patents_from_uspto
from src.db import PatentVectorStore

_EMBED_MODEL = None

def get_embedding_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print("[Ingest] Loading base embedding model BAAI/bge-small-en-v1.5...")
        _EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _EMBED_MODEL

# Public singleton alias — used by rag.py's semantic floor without reloading model
get_embedder = get_embedding_model

def generate_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def ingest_patents_for_keywords(keywords: list[str], vector_store: PatentVectorStore, limit: int = 15):
    """
    Fetches patent data and ingests generated claim embeddings into the vector store.
    """
    raw_patents = fetch_patents_from_uspto(keywords, limit=limit)
    print(f"[Ingest] Processing and embedding {len(raw_patents)} patent records...")

    for patent in raw_patents:
        p_num = patent.get("patent_number", "US-UNKNOWN")
        p_title = patent.get("patent_title", "Untitled Patent")
        p_abstract = patent.get("patent_abstract", "")
        assignee = patent.get("assignee_organization") or "Independent Inventor"
        p_date = patent.get("patent_date", "2024-01-01")

        claim_text = f"Patent Title: {p_title}. Claim Text: {p_abstract}"
        embedding = generate_embedding(claim_text)

        vector_store.insert_claim(
            patent_number=p_num,
            patent_title=p_title,
            claim_text=claim_text,
            assignee=assignee,
            patent_date=p_date,
            embedding=embedding
        )

    print(f"[Ingest] Successfully ingested {len(raw_patents)} patent claims!")

if __name__ == "__main__":
    store = PatentVectorStore()
    ingest_patents_for_keywords(["neural", "vector"], store, limit=5)
