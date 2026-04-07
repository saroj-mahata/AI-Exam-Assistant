import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# --- Configure Gemini ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Please check your .env file.")
genai.configure(api_key=api_key)

app = FastAPI(title="AI Exam Assistant", version="3.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model ---
model = genai.GenerativeModel("gemini-2.0-flash")

# --- In-memory state ---
uploaded_pdfs: dict[str, str] = {}
chat_history: List[dict] = []

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
MAX_CONTEXT_CHARS   = 40_000


# --- Schemas ---
class ChatRequest(BaseModel):
    message: str

class RemovePdfRequest(BaseModel):
    filename: str


# --- Helpers ---
def get_combined_context() -> str:
    if not uploaded_pdfs:
        return ""
    parts = [f"=== {fname} ===\n{text}" for fname, text in uploaded_pdfs.items()]
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def build_system_prompt() -> str:
    context = get_combined_context()
    if context:
        return (
            "You are an expert AI study assistant. You have the student's uploaded study notes below.\n\n"
            "RULES:\n"
            "1. If the question can be answered from the notes, answer using the notes and briefly say so.\n"
            "2. If the question is NOT in the notes but is a valid academic or general knowledge topic, "
            "answer from your own knowledge and add: '(This answer is from general knowledge, not your notes)'\n"
            "3. If a question is completely unrelated to studying or academics, politely redirect.\n"
            "4. Be concise, clear, and helpful like a knowledgeable tutor.\n"
            "5. You have memory of the conversation — use it for coherent follow-up answers.\n\n"
            f"--- STUDENT NOTES ---\n{context}\n--- END OF NOTES ---"
        )
    return (
        "You are an expert AI study assistant. No notes have been uploaded yet.\n"
        "Answer questions from your general knowledge as a knowledgeable tutor.\n"
        "Encourage the user to upload PDF notes for personalized answers.\n"
        "Be concise, clear, and helpful."
    )


# --- Routes ---

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0"}


@app.get("/status")
def get_status():
    return {
        "pdfs": [{"filename": f, "characters": len(t)} for f, t in uploaded_pdfs.items()],
        "total_characters": sum(len(t) for t in uploaded_pdfs.values()),
        "chat_turns": len(chat_history),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max size is 5 MB.")

    filename = file.filename
    if filename in uploaded_pdfs:
        raise HTTPException(status_code=409, detail=f"'{filename}' is already uploaded. Remove it first to replace.")

    try:
        reader = PdfReader(io.BytesIO(contents))
        extracted = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted += text + "\n"

        if not extracted.strip():
            raise HTTPException(status_code=422, detail="No text found. This PDF may be scanned or image-based.")

        uploaded_pdfs[filename] = extracted

        return {
            "message": f"'{filename}' uploaded successfully.",
            "filename": filename,
            "pages": len(reader.pages),
            "characters": len(extracted),
            "total_pdfs": len(uploaded_pdfs),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.delete("/remove-pdf")
async def remove_pdf(body: RemovePdfRequest):
    if body.filename not in uploaded_pdfs:
        raise HTTPException(status_code=404, detail=f"'{body.filename}' not found.")
    del uploaded_pdfs[body.filename]
    return {"message": f"'{body.filename}' removed.", "remaining": list(uploaded_pdfs.keys())}


@app.delete("/clear-pdfs")
async def clear_pdfs():
    uploaded_pdfs.clear()
    return {"message": "All PDFs cleared."}


@app.post("/chat")
async def chat(body: ChatRequest):
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    system_prompt = build_system_prompt()

    gemini_history = []
    for turn in chat_history:
        role = "user" if turn["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [turn["content"]]})

    try:
        convo = model.start_chat(history=gemini_history)
        full_message = f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n\n[USER]\n{message}"
        response = convo.send_message(full_message)
        reply = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": reply})

    while len(chat_history) > 30:
        chat_history.pop(0)
        chat_history.pop(0)

    return {"reply": reply, "has_notes": bool(uploaded_pdfs)}


@app.delete("/clear-chat")
async def clear_chat():
    chat_history.clear()
    return {"message": "Chat cleared."}


@app.get("/weak-topics")
async def get_weak_topics():
    if not uploaded_pdfs:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF first.")
    prompt = (
        "You are an experienced teacher. Analyze the study notes below and identify "
        "the top 3 most difficult topics a student is likely to struggle with.\n"
        "For each:\n1. Topic name\n2. Why it's hard (1-2 sentences)\n3. One study tip\n\n"
        f"Notes:\n{get_combined_context()}"
    )
    try:
        return {"weak_topics": model.generate_content(prompt).text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


@app.get("/generate-test")
async def generate_test():
    if not uploaded_pdfs:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF first.")
    prompt = (
        "You are an exam setter. Based strictly on the notes below, create:\n"
        "- 3 multiple-choice questions (4 options: A/B/C/D)\n"
        "- 2 short-answer questions\n\n"
        "End with a '--- ANSWER KEY ---' section.\n\n"
        f"Notes:\n{get_combined_context()}"
    )
    try:
        return {"test": model.generate_content(prompt).text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")