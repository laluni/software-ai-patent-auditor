import os
import json
import numpy as np

# Try importing psycopg2 / pgvector; fallback to in-memory store if DB isn't running yet
try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    HAS_PG = True
except ImportError:
    HAS_PG = False

class PatentVectorStore:
    def __init__(self, host="localhost", port=5432, dbname="patent_db", user="postgres", password="postgres_password"):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.conn = None
        self.use_pg = False

        # In-memory storage fallback
        self.in_memory_docs = []
        self.in_memory_embeddings = []

        self.connect()

    def connect(self):
        if HAS_PG:
            try:
                self.conn = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    dbname=self.dbname,
                    user=self.user,
                    password=self.password
                )
                self.conn.autocommit = True
                register_vector(self.conn)
                self.use_pg = True
                print("[VectorStore] Connected to PostgreSQL + PGVector successfully!")
                self._init_db_schema()
                return
            except Exception as e:
                print(f"[VectorStore Warning] Could not connect to PostgreSQL: {e}")
                print("[VectorStore] Falling back to In-Memory Vector & Lexical Store.")
        
        self.use_pg = False

    def _init_db_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS patent_claims (
                    id SERIAL PRIMARY KEY,
                    patent_number VARCHAR(50),
                    patent_title TEXT,
                    claim_text TEXT,
                    assignee VARCHAR(255),
                    patent_date VARCHAR(20),
                    embedding vector(384),
                    fts_doc tsvector
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS patent_claims_fts_idx ON patent_claims USING gin(fts_doc);")

    def insert_claim(self, patent_number: str, patent_title: str, claim_text: str, assignee: str, patent_date: str, embedding: list[float]):
        if self.use_pg:
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO patent_claims (patent_number, patent_title, claim_text, assignee, patent_date, embedding, fts_doc)
                        VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('english', %s));
                    """, (patent_number, patent_title, claim_text, assignee, patent_date, embedding, claim_text))
                return
            except Exception as e:
                print(f"[VectorStore Exception] PG insert error: {e}. Storing in memory.")

        # In-memory fallback
        doc = {
            "patent_number": patent_number,
            "patent_title": patent_title,
            "claim_text": claim_text,
            "assignee": assignee,
            "patent_date": patent_date
        }
        self.in_memory_docs.append(doc)
        self.in_memory_embeddings.append(np.array(embedding, dtype=np.float32))

    def hybrid_search(self, query_text: str, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Executes Hybrid Search (Dense Cosine Similarity + Sparse Lexical BM25/FTS) with RRF.
        """
        if self.use_pg:
            try:
                return self._pg_hybrid_search(query_text, query_embedding, top_k)
            except Exception as e:
                print(f"[VectorStore] PG Hybrid search error: {e}. Falling back to memory.")

        return self._in_memory_hybrid_search(query_text, query_embedding, top_k)

    def _pg_hybrid_search(self, query_text: str, query_embedding: list[float], top_k: int) -> list[dict]:
        with self.conn.cursor() as cur:
            # 1. Dense Vector Search Top 20
            cur.execute("""
                SELECT patent_number, patent_title, claim_text, assignee, patent_date, 
                       (embedding <=> %s::vector) AS distance
                FROM patent_claims
                ORDER BY distance ASC
                LIMIT 20;
            """, (query_embedding,))
            dense_results = cur.fetchall()

            # 2. Sparse Lexical FTS Search Top 20
            cur.execute("""
                SELECT patent_number, patent_title, claim_text, assignee, patent_date,
                       ts_rank(fts_doc, plainto_tsquery('english', %s)) AS rank
                FROM patent_claims
                WHERE fts_doc @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT 20;
            """, (query_text, query_text))
            sparse_results = cur.fetchall()

        # Reciprocal Rank Fusion (RRF)
        return self._rrf_fusion(dense_results, sparse_results, top_k)

    def _in_memory_hybrid_search(self, query_text: str, query_embedding: list[float], top_k: int) -> list[dict]:
        if not self.in_memory_docs:
            return []

        # Cosine Distance Computation
        query_vec = np.array(query_embedding, dtype=np.float32)
        norm_q = np.linalg.norm(query_vec)
        
        dense_ranks = []
        for i, doc_vec in enumerate(self.in_memory_embeddings):
            norm_d = np.linalg.norm(doc_vec)
            sim = np.dot(query_vec, doc_vec) / (norm_q * norm_d + 1e-8)
            dense_ranks.append((i, 1.0 - sim)) # Distance = 1 - Cosine Sim

        dense_ranks.sort(key=lambda x: x[1])

        # Lexical Matching Score
        keywords = set(query_text.lower().split())
        sparse_ranks = []
        for i, doc in enumerate(self.in_memory_docs):
            match_count = sum(1 for word in keywords if word in doc["claim_text"].lower())
            sparse_ranks.append((i, match_count))

        sparse_ranks.sort(key=lambda x: x[1], reverse=True)

        # RRF Fusion
        rrf_scores = {}
        k = 60
        for rank, (idx, _) in enumerate(dense_ranks[:20]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        for rank, (idx, match) in enumerate(sparse_ranks[:20]):
            if match > 0:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in sorted_indices:
            doc = self.in_memory_docs[idx].copy()
            doc["rrf_score"] = round(score, 4)
            results.append(doc)

        return results

    def _rrf_fusion(self, dense_rows, sparse_rows, top_k):
        k = 60
        scores = {}
        docs = {}

        for rank, row in enumerate(dense_rows):
            p_num = row[0]
            scores[p_num] = scores.get(p_num, 0.0) + 1.0 / (k + rank + 1)
            docs[p_num] = {
                "patent_number": row[0],
                "patent_title": row[1],
                "claim_text": row[2],
                "assignee": row[3],
                "patent_date": row[4]
            }

        for rank, row in enumerate(sparse_rows):
            p_num = row[0]
            scores[p_num] = scores.get(p_num, 0.0) + 1.0 / (k + rank + 1)
            if p_num not in docs:
                docs[p_num] = {
                    "patent_number": row[0],
                    "patent_title": row[1],
                    "claim_text": row[2],
                    "assignee": row[3],
                    "patent_date": row[4]
                }

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for p_num, score in sorted_items:
            doc = docs[p_num].copy()
            doc["rrf_score"] = round(score, 4)
            results.append(doc)

        return results
