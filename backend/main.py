from fastapi import FastAPI, UploadFile, File
import os
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Latest Fast Model
model = genai.GenerativeModel("gemini-1.5-flash")

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

            reader = PdfReader(file_path)

            for page in reader.pages:

                content = page.extract_text()

                if content:
                    text += content

    return text


# Ask Question
@app.post("/ask")
async def ask_question(question: str):

    user_questions.append(question)

    notes = read_all_pdfs()

    prompt = f"""
    You are an AI Exam Assistant.

    Study Notes:
    {notes}

    Question:
    {question}

    Answer clearly and concisely.
    """

    response = model.generate_content(prompt)

    return {"answer": response.text}


# Weak Topic Detection
@app.get("/weak-topics")
async def weak_topics():

    questions = "\n".join(user_questions)

    prompt = f"""
    Based on these student questions:

    {questions}

    Identify weak topics and suggest improvement areas.
    """

    response = model.generate_content(prompt)

    return {"weak_topics": response.text}


# Generate Test
@app.get("/generate-test")
async def generate_test():

    notes = read_all_pdfs()

    prompt = f"""
    Generate 5 exam questions from these notes:

    {notes}

    Format:
    1.
    2.
    3.
    4.
    5.
    """

    response = model.generate_content(prompt)

    return {"test": response.text}
