from fastapi import FastAPI, UploadFile, File
import os
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize model
model = genai.GenerativeModel("gemini-pro")

app = FastAPI()

UPLOAD_FOLDER = "data"

# Store user questions
user_questions = []


@app.get("/")
def home():
    return {"message": "AI Exam Assistant Running"}


# Upload PDF
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    return {"message": "File uploaded successfully"}


# Read PDFs
def read_all_pdfs():
    text = ""
    
    if not os.path.exists(UPLOAD_FOLDER):
        return text
    
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".pdf"):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
    
    return text


# Ask Question
@app.post("/ask")
async def ask_question(question: str):
    user_questions.append(question)

    content = read_all_pdfs()

    prompt = f"""
    Answer from these notes:
    
    {content}
    
    Question:
    {question}
    """

    response = model.generate_content(prompt)

    return {"answer": response.text}


# Weak Topics
@app.get("/weak-topics")
async def weak_topics():
    questions = "\n".join(user_questions)

    prompt = f"""
    Based on these user questions:
    
    {questions}
    
    Identify weak topics of the student.
    """

    response = model.generate_content(prompt)

    return {"weak_topics": response.text}


# Generate Test
@app.get("/generate-test")
async def generate_test():
    content = read_all_pdfs()

    prompt = f"""
    Generate 5 exam questions from these notes:

    {content}

    Format:
    1.
    2.
    3.
    4.
    5.
    """

    response = model.generate_content(prompt)

    return {"test": response.text}
