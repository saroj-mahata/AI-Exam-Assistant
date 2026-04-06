from fastapi import FastAPI, UploadFile, File
import os
import PyPDF2

app = FastAPI()

UPLOAD_FOLDER = "data"

@app.get("/")
def home():
    return {"message": "AI Exam Assistant Running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    return {"message": "File uploaded successfully"}


def read_all_pdfs():
    text = ""
    
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".pdf"):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
    
    return text


@app.post("/ask")
async def ask_question(question: str):
    content = read_all_pdfs()

    if question.lower() in content.lower():
        return {"answer": "Answer found in your notes"}
    
    return {"answer": "I couldn't find answer in uploaded PDFs"}
