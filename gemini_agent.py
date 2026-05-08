# agents/gemini_agent.py
import os
from typing import List, Dict, Any
from utils.pdf_loader import load_pdf_text
from utils.text_splitter import simple_split
from utils.vectordb import SimpleVectorDB
from dotenv import load_dotenv
import google.generativeai as genai
import json
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment. See .env.example")

# configure client
genai.configure(api_key=GEMINI_KEY)

# model is configurable in .env, fallback to gemini-1.5
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5")

def _call_gemini_system(messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
    """
    Call Gemini chat API with messages.
    messages: list of {"role": "system/user/assistant", "content": "..."}
    Returns text content.
    """
    resp = genai.chat.create(
        model=GEMINI_MODEL,
        messages=messages,
        max_output_tokens=max_tokens,
        temperature=0.0
    )
    # the API returns structured content — adapt to returned shape
    # `resp` contains `candidates` with `content` typically in `resp.candidates[0].content`.
    # google.generativeai library returns an object with `candidates` and nested `content` pieces.
    try:
        # Flatten text pieces if needed
        content = ""
        for candidate in resp.candidates:
            # candidate may have "content" with list of parts
            c = candidate.get("content")
            if isinstance(c, list):
                for item in c:
                    if item.get("type") == "output_text":
                        content += item.get("text", "")
            elif isinstance(c, str):
                content += c
        if not content:
            # fallback: try string conversion
            content = str(resp)
        return content.strip()
    except Exception:
        return str(resp)

class GeminiAgent:
    def __init__(self):
        self.vdb = SimpleVectorDB()
        self.n_chunks = 0

    def ingest_pdf(self, path: str) -> Dict[str, Any]:
        text = load_pdf_text(path)
        chunks = simple_split(text, chunk_size=1000, overlap=200)
        self.vdb.create_index(chunks)
        self.n_chunks = len(chunks)
        return {"n_chunks": self.n_chunks}

    def summarize(self) -> str:
        # use first few chunks as representative context
        k = min(4, self.n_chunks)
        example_context = "\n\n".join([self.vdb.get_text(i) for i in range(k)]) if k>0 else ""
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that summarizes technical documents."},
            {"role": "user", "content": f"Provide a concise technical summary in 6 bullet points for the following document excerpt:\n\n{example_context}"}
        ]
        return _call_gemini_system(prompt, max_tokens=400)

    def classify(self, labels: List[str] = None):
        if labels is None:
            labels = ["Medical", "Mechanical", "Electrical", "Chemical", "Software/AI", "Other"]
        sample = "\n\n".join([self.vdb.get_text(i) for i in range(min(5, self.n_chunks))])
        prompt = [
            {"role": "system", "content": "You are an expert classifier. Output valid JSON."},
            {"role": "user", "content": f"Classify the following document excerpt into the labels {labels}. "
                                       f"Return a JSON object mapping label -> probability (0-1) or a single label if applicable.\n\nDocument excerpt:\n{sample}"}
        ]
        out = _call_gemini_system(prompt, max_tokens=200)
        # try to parse JSON
        try:
            return json.loads(out)
        except:
            return {"raw": out}

    def extract_citations(self):
        citations = []
        for i in range(self.n_chunks):
            chunk = self.vdb.get_text(i)
            prompt = [
                {"role": "system", "content": "You are an expert in extracting references, patent numbers, figure citations from technical text."},
                {"role": "user", "content": f"Extract any citation-like items (patent numbers, references, figure references, bibliographic items) from this chunk and return a JSON array of objects with fields 'chunk_id' and 'text'. If none, return []:\n\n{chunk}"}
            ]
            out = _call_gemini_system(prompt, max_tokens=200)
            try:
                arr = json.loads(out)
                if isinstance(arr, list):
                    for a in arr:
                        a["chunk_id"] = i
                        citations.append(a)
            except:
                # fallback heuristic
                low = chunk.lower()
                if "patent" in low or "fig." in low or "reference" in low or "doi" in low:
                    citations.append({"chunk_id": i, "text": chunk[:300]})
        return citations

    def answer_question(self, question: str, top_k: int = 5) -> str:
        hits = self.vdb.query(question, top_k=top_k)
        if not hits:
            # if no index, fallback to whole text
            context = ""
        else:
            contexts = []
            for idx, dist in hits:
                contexts.append(f"CHUNK[{idx}] (dist={dist}):\n{self.vdb.get_text(idx)}")
            context = "\n\n".join(contexts)
        prompt = [
            {"role": "system", "content": "You are a helpful assistant that answers questions using the provided context. Mention SOURCE: CHUNK[index] for citations."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and cite sources like SOURCE: CHUNK[2]."}
        ]
        return _call_gemini_system(prompt, max_tokens=400)

def get_agent():
    return GeminiAgent()
