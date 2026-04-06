import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load Environment Variables (Your .env file stays hidden!)
load_dotenv()

# 2. Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("🚨 GOOGLE_API_KEY is missing. Please create a .env file locally.")

genai.configure(api_key=api_key)

# 3. Initialize FastAPI App
app = FastAPI(title="AI Exam Assistant API")

# 4. Configure CORS (CRITICAL: Allows frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini model (1.5-flash is fast and great for text)
model = genai.GenerativeModel('gemini-1.5-flash')

@app.get("/")
async def health_check():
    return {"message": "AI Exam Assistant Backend is running smoothly!"}

@app.post("/analyze-pdf")
async def analyze_exam_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF, extracts text, and uses Gemini to analyze it.
    """
    # Check if the uploaded file is actually a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        # 5. Extract Text from PDF using modern pypdf
        reader = PdfReader(file.file)
        extracted_text = ""
        for page in reader.pages:
            # Extract text and add a newline character after each page
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"

        if not extracted_text.strip():
             raise HTTPException(status_code=400, detail="Could not read text from this PDF. It might be scanned or image-based.")

        # 6. Create the Prompt for Gemini
        prompt = f"""
        You are an expert AI Exam Assistant. Analyze the following text extracted from a student's study material.
        
        Please provide your response in these 3 clear sections:
        1. **Summary**: A brief summary of the main subject.
        2. **Focus Areas**: Top 3 crucial/weak topics the student must focus on based on this material.
        3. **Practice Quiz**: 3 sample practice questions to test their knowledge.
        
        --- EXAM MATERIAL TEXT ---
        {extracted_text[:15000]} 
        """
        # Note: We slice the text [:15000] to ensure we don't send a massive book 
        # all at once and hit rate limits during testing.

        # 7. Call Gemini API
        response = model.generate_content(prompt)

        # 8. Return data to the frontend
        return {
            "filename": file.filename,
            "status": "success",
            "ai_analysis": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing: {str(e)}")
