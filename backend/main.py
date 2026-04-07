import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# --- Configure Gemini ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Please check your .env file.")
genai.configure(api_key=api_key)

app = FastAPI(title="AI Exam Assistant", version="2.0")

# --- CORS ---
# For production, replace the origins list with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model ---
model = genai.GenerativeModel("gemini-2.0-flash")

# --- In-memory state ---
# Note: This is single-user only. For multi-user, use a session store or database.
current_pdf_text: str = ""
current_filename: str = ""

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_CONTEXT_CHARS = 30_000


# --- Request body schema ---
class QuestionRequest(BaseModel):
    question: str


# --- Helpers ---
def get_pdf_context() -> str:
    """Returns truncated PDF text for use in prompts."""
    return current_pdf_text[:MAX_CONTEXT_CHARS]


def require_pdf():
    """Raises 400 if no PDF has been uploaded yet."""
    if not current_pdf_text:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")


# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0"}


@app.get("/status")
def get_status():
    return {
        "pdf_loaded": bool(current_pdf_text),
        "filename": current_filename or None,
        "characters": len(current_pdf_text),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global current_pdf_text, current_filename

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    try:
        import io
        reader = PdfReader(io.BytesIO(contents))

        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from this PDF. It may be scanned or image-based.",
            )

        current_pdf_text = extracted_text
        current_filename = file.filename

        return {
            "message": f"Successfully read {len(current_pdf_text):,} characters from '{file.filename}'. You can now ask questions!",
            "characters": len(current_pdf_text),
            "pages": len(reader.pages),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/ask")
async def ask_question(body: QuestionRequest):
    require_pdf()

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    prompt = (
        f"You are a helpful study assistant. Using ONLY the notes provided below, "
        f"answer the following question as clearly and concisely as possible.\n\n"
        f"Question: {question}\n\n"
        f"Notes:\n{get_pdf_context()}"
    )

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


@app.get("/weak-topics")
async def get_weak_topics():
    require_pdf()

    prompt = (
        "You are an experienced teacher. Analyze the study notes below and identify "
        "the top 3 most difficult or complex topics that a student is likely to struggle with. "
        "For each topic:\n"
        "1. State the topic name clearly\n"
        "2. Explain in 1-2 sentences why it is challenging\n"
        "3. Give one tip for how to study it effectively\n\n"
        f"Notes:\n{get_pdf_context()}"
    )

    try:
        response = model.generate_content(prompt)
        return {"weak_topics": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")


@app.get("/generate-test")
async def generate_practice_test():
    require_pdf()

    prompt = (
        "You are an exam setter. Based strictly on the notes provided below, "
        "create a well-structured practice test with:\n"
        "- 3 multiple-choice questions (with 4 options each, label them A/B/C/D)\n"
        "- 2 short-answer questions\n\n"
        "After all questions, include a clearly separated '--- ANSWER KEY ---' section "
        "with the correct answers.\n\n"
        f"Notes:\n{get_pdf_context()}"
    )

    try:
        response = model.generate_content(prompt)
        return {"test": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")
