# utils/pdf_loader.py
import pdfplumber
from typing import List

def load_pdf_text(path: str) -> str:
    """
    Load all text from a PDF and return a single string.
    """
    text_pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_pages.append(text)
    return "\n\n".join(text_pages)
