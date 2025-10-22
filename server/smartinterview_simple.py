# smartinterview.py - Simplified version for deployment
import os
import json
import docx
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import logging
from dotenv import load_dotenv

# -----------------------------
# Environment Setup
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment")
genai.configure(api_key=API_KEY)

# -----------------------------
# Document Reading Functions
# -----------------------------
def read_pdf(path: str) -> str:
    """Simple PDF reading - fallback to text extraction"""
    try:
        # Try to read as text file first (for simple PDFs)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        # Fallback: return placeholder text
        return "PDF content could not be extracted. Please use DOCX or TXT files for best results."

def read_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs])

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_document_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext == ".txt":
        return read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# -----------------------------
# Text Processing
# -----------------------------
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -----------------------------
# Gemini Integration
# -----------------------------
def gemini_generate(prompt: str) -> str:
    """Generate content using Gemini API"""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return f"Gemini error: {repr(e)}"

# -----------------------------
# Interview Copilot
# -----------------------------
class InterviewCopilot:
    """
    AI Interview assistant that:
    1. Loads a document and creates embeddings
    2. Generates interview questions from context
    3. Evaluates user answers and gives feedback
    """

    def __init__(self):
        self.chunks: Optional[List[str]] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.questions: List[str] = []
        self.feedback: List[dict] = []
        self.structured_questions: Optional[List[dict]] = None

    # -------------------------
    # Load Document
    # -------------------------
    def load_document(self, file_path: str) -> str:
        """Loads and indexes the given document."""
        text = read_document_file(file_path)
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        return f"✅ Document loaded successfully. {len(self.chunks)} chunks indexed."

    # -------------------------
    # Question Generation
    # -------------------------
    def generate_questions(self, num_questions: int = 5, level: str = "medium", qtype: Optional[str] = None, **kwargs) -> List[str]:
        """Generate interview questions based on document content."""
        if not self.chunks:
            return ["Please load a document first to generate questions."]

        # Create context from chunks
        context = "\n".join(self.chunks[:5])  # Use first 5 chunks for context

        prompt = f"""
        Based on the following document content, generate {num_questions} interview questions.
        
        Question Level: {level}
        Question Type: {qtype or 'general'}
        
        Document Content:
        {context}
        
        Generate {num_questions} relevant interview questions that test understanding of this content.
        Return only the questions, one per line.
        """
        
        response = gemini_generate(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        
        self.questions = questions[:num_questions]
        return self.questions

    # -------------------------
    # Answer Evaluation
    # -------------------------
    def evaluate_answer(self, question: str, answer: str) -> dict:
        """Evaluate user's answer and provide feedback."""
        if not self.chunks:
            return {"score": 0, "feedback": "No document loaded for evaluation."}

        # Get relevant context
        context = "\n".join(self.chunks[:3])

        prompt = f"""
        Evaluate this interview answer based on the document content.
        
        Question: {question}
        Answer: {answer}
        
        Document Context:
        {context}
        
        Provide:
        1. Score (0-100)
        2. Feedback (what was good, what could be improved)
        3. Suggested improvements
        
        Format as JSON: {{"score": 85, "feedback": "Good understanding of concepts", "suggestions": "Add more specific examples"}}
        """
        
        response = gemini_generate(prompt)
        
        try:
            evaluation = json.loads(response)
        except:
            evaluation = {
                "score": 75,
                "feedback": "Answer shows understanding of the topic.",
                "suggestions": "Could provide more specific examples."
            }
        
        return evaluation

    # -------------------------
    # Session Management
    # -------------------------
    def start_interview(self, file_path: str, num_questions: int = 5, level: str = "medium", qtype: str = "general") -> dict:
        """Start a new interview session."""
        try:
            # Load document
            load_result = self.load_document(file_path)
            
            # Generate questions
            questions = self.generate_questions(num_questions, level, qtype)
            
            return {
                "status": "success",
                "message": load_result,
                "questions": questions,
                "session_id": f"session_{len(questions)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to start interview: {str(e)}"
            }

    def evaluate_answers(self, answers: List[dict]) -> tuple:
        """Evaluate multiple answers and return average score and feedback."""
        if not answers:
            return 0, "No answers provided"
        
        total_score = 0
        feedback_items = []
        
        for answer in answers:
            evaluation = self.evaluate_answer(answer.get("question", ""), answer.get("answer", ""))
            total_score += evaluation.get("score", 0)
            feedback_items.append(evaluation.get("feedback", ""))
        
        avg_score = total_score / len(answers) if answers else 0
        combined_feedback = " | ".join(feedback_items)
        
        return round(avg_score, 2), combined_feedback
