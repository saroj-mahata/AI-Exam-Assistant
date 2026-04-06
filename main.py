import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Please check your .env file.")
genai.configure(api_key=api_key)

app = FastAPI()

# Allow Frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = genai.GenerativeModel('gemini-1.5-flash')

# --- SIMPLE MEMORY ---
# This stores the PDF text so the other endpoints can use it.
# Note: In a real production app with multiple users, you would use a database.
current_pdf_text = "" 

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global current_pdf_text
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        reader = PdfReader(file.file)
        extracted_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        
        current_pdf_text = extracted_text
        return {"message": f"Successfully read {len(current_pdf_text)} characters from {file.filename}. You can now ask questions!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: str):
    global current_pdf_text
    if not current_pdf_text:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
    
    prompt = f"Based on the following notes, answer this question: '{question}'\n\nNotes:\n{current_pdf_text[:15000]}"
    response = model.generate_content(prompt)
    return {"answer": response.text}

@app.get("/weak-topics")
async def get_weak_topics():
    global current_pdf_text
    if not current_pdf_text:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
    
    prompt = f"Analyze these notes and list the top 3 most difficult or complex topics that a student might struggle with. Briefly explain why.\n\nNotes:\n{current_pdf_text[:15000]}"
    response = model.generate_content(prompt)
    return {"weak_topics": response.text}

@app.get("/generate-test")
async def generate_practice_test():
    global current_pdf_text
    if not current_pdf_text:
        raise HTTPException(status_code=400, detail="Please upload a PDF first.")
    
    prompt = f"Create a short practice test with 3 multiple-choice questions and 2 short-answer questions based strictly on these notes. Provide the answer key at the very bottom.\n\nNotes:\n{current_pdf_text[:15000]}"
    response = model.generate_content(prompt)
    return {"test": response.text}
