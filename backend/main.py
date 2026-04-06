from fastapi import FastAPI, UploadFile, File
import os

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
