from src.db import PatentVectorStore
from src.ingest import generate_embedding, ingest_patents_for_keywords

class SearchEngine:
    def __init__(self, vector_store: PatentVectorStore = None):
        self.store = vector_store or PatentVectorStore()

    def search_prior_art(self, design_doc: str, query_keywords: list[str] = None, top_k: int = 5) -> list[dict]:
        """
        Performs Hybrid Search across indexed patents for a given developer design doc.
        Automatically triggers ingestion if index is empty.
        """
        if query_keywords and not self.store.in_memory_docs and not self.store.use_pg:
            print("[SearchEngine] Initializing index with keyword search data...")
            ingest_patents_for_keywords(query_keywords, self.store, limit=15)

        # Generate dense query embedding
        query_embedding = generate_embedding(design_doc)
        
        # Execute hybrid search
        results = self.store.hybrid_search(
            query_text=design_doc,
            query_embedding=query_embedding,
            top_k=top_k
        )
        return results

if __name__ == "__main__":
    engine = SearchEngine()
    results = engine.search_prior_art("System for vector search", query_keywords=["vector", "search"])
    print("Found Results Count:", len(results))
