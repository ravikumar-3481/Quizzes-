"""
Intelligent MCQ Extraction API
------------------------------
Requirements to run:
pip install fastapi uvicorn python-multipart torch sentence-transformers pydantic

To run the server:
uvicorn backend_api:app --reload --port 8000
"""

import re
import io
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Machine Learning & Deep Learning Imports ---
import torch
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="ML MCQ Quiz API")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Deep Learning model for semantic similarity (runs on CPU or GPU)
# We use this to detect and remove duplicate questions across multiple uploaded files.
print("Loading Deep Learning model for semantic deduplication...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print(f"Model loaded successfully on {device.upper()}.")
except Exception as e:
    print(f"Warning: Could not load DL model. Semantic deduplication will be disabled. Error: {e}")
    embedding_model = None


class MCQResponse(BaseModel):
    total_parsed: int
    total_unique: int
    questions: List[Dict[str, Any]]


def parse_mcq_text(text: str) -> List[Dict[str, Any]]:
    """
    Robust text parser to extract MCQs from raw text files.
    Expected format:
    1. What is the capital of France?
    A) London
    B) Paris
    C) Berlin
    D) Rome
    Answer: B
    """
    questions = []
    
    # Split text into blocks based on numbers followed by a dot (e.g., "1.", "2.")
    blocks = re.split(r'\n(?=\d+[\.\)])', text)
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue
            
        # The first line is usually the question
        question_text = re.sub(r'^\d+[\.\)]\s*', '', lines[0]).strip()
        options = {}
        correct_answer = None
        
        for line in lines[1:]:
            # Match options like "A) London" or "a. London"
            opt_match = re.match(r'^([a-d])[\.\)]\s*(.*)', line, re.IGNORECASE)
            if opt_match:
                opt_letter = opt_match.group(1).upper()
                opt_text = opt_match.group(2).strip()
                options[opt_letter] = opt_text
                continue
                
            # Match answers like "Answer: A" or "Ans: A"
            ans_match = re.match(r'^(?:answer|ans)[\s:]*([a-d])', line, re.IGNORECASE)
            if ans_match:
                correct_answer = ans_match.group(1).upper()
        
        if question_text and len(options) >= 2 and correct_answer in options:
            questions.append({
                "question": question_text,
                "options": options,
                "answer": correct_answer
            })
            
    return questions

def remove_duplicates_dl(questions: List[Dict[str, Any]], threshold: float = 0.92) -> List[Dict[str, Any]]:
    """
    Uses Deep Learning (Sentence Embeddings) to identify and remove 
    semantically duplicate questions across different uploaded files.
    """
    if not embedding_model or not questions:
        return questions

    texts = [q["question"] for q in questions]
    
    # Generate dense vector embeddings for all questions
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    
    # Compute cosine similarity matrix
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    unique_questions = []
    seen_indices = set()
    
    for i in range(len(questions)):
        if i in seen_indices:
            continue
            
        unique_questions.append(questions[i])
        seen_indices.add(i)
        
        # Find all questions that are highly similar to question `i`
        for j in range(i + 1, len(questions)):
            if j not in seen_indices and cosine_scores[i][j].item() > threshold:
                # Mark as seen (duplicate)
                seen_indices.add(j)
                
    return unique_questions


@app.post("/api/upload", response_model=MCQResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Endpoint to receive multiple files, extract questions, 
    and apply ML-based deduplication.
    """
    all_extracted_questions = []
    
    for file in files:
        if not file.filename.endswith(('.txt', '.md', '.csv')):
            # For simplicity, we process text files.
            # In a full production app, you could add PyPDF2 to handle PDFs.
            continue
            
        content = await file.read()
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
            
        file_questions = parse_mcq_text(text)
        all_extracted_questions.extend(file_questions)
        
    total_parsed = len(all_extracted_questions)
    
    # Apply ML deduplication
    unique_questions = remove_duplicates_dl(all_extracted_questions)
    total_unique = len(unique_questions)
    
    if total_unique == 0:
        raise HTTPException(status_code=400, detail="No valid MCQs could be extracted from the provided files.")
        
    return {
        "total_parsed": total_parsed,
        "total_unique": total_unique,
        "questions": unique_questions
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "ml_model_loaded": embedding_model is not None}
