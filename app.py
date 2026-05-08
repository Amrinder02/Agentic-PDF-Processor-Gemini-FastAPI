# app.py
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import shutil
from dotenv import load_dotenv
load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from agents.gemini_agent import get_agent

agent = get_agent()

app = FastAPI(title="Agentic PDF Processor (Gemini)")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")
    out_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(out_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    result = agent.ingest_pdf(out_path)
    return JSONResponse({"status": "ok", "ingest_result": result})

@app.get("/summary")
def summary():
    s = agent.summarize()
    return {"summary": s}

@app.post("/classify")
def classify(labels: str = Form(None)):
    label_list = [l.strip() for l in labels.split(",")] if labels else None
    parsed = agent.classify(label_list)
    return {"classification": parsed}

@app.get("/citations")
def citations():
    c = agent.extract_citations()
    return {"citations": c}

@app.post("/query")
def query(question: str = Form(...)):
    ans = agent.answer_question(question)
    return {"answer": ans}

@app.get("/")
def root():
    return {"message": "Agentic PDF Processor (Gemini). Use /docs for API UI."}
