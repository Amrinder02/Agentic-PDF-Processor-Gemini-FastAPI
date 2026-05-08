# utils/text_splitter.py
from typing import List

def simple_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple character-based splitter with overlap.
    Returns list of text chunks.
    """
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks
