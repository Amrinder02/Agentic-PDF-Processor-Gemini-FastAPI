# Agentic PDF Processor — Gemini + FastAPI

This project ingests PDFs, builds a vector DB of text chunks, and exposes an agentic workflow using Google's Gemini models (via `google-generativeai`) to:

- Summarize the document
- Classify into labels
- Extract citation-like references
- Answer context-aware questions (RAG)

## Setup

1. Clone repo and `cd` into it.

2. Create virtualenv and install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
