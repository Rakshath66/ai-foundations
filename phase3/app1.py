from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

app = FastAPI()

# Dummy placeholder for chatbot response logic
def generate_answer_from_pdf(pdf_path: str, question: str) -> str:
    return f"Answer to: '{question}' based on {os.path.basename(pdf_path)}"

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Upload a PDF and get an answer
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), question: str = Form(...)):
    # Save uploaded file to disk
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Generate dummy answer
    answer = generate_answer_from_pdf(file_path, question)
    return {"filename": file.filename, "answer": answer}
