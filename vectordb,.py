# utils/vectordb.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple

class SimpleVectorDB:
    """
    Minimal FAISS-backed vector DB that stores chunks and embeddings in memory.
    Suitable for demos and GitHub projects.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.embeddings = None

    def create_index(self, texts: List[str]):
        self.texts = texts
        if len(texts) == 0:
            self.index = None
            self.embeddings = None
            return
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)
        self.embeddings = embeddings

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.index is None:
            return []
        q_emb = self.embed_model.encode([query_text], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        distances = D[0].tolist()
        indices = I[0].tolist()
        return list(zip(indices, distances))

    def get_text(self, idx: int) -> str:
        return self.texts[idx]
