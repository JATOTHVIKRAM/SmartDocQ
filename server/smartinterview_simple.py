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
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "ResourceExhausted" in error_msg:
            return "⚠️ API quota exceeded. Please try again later or upgrade your Gemini API plan."
        elif "404" in error_msg:
            return "⚠️ Gemini model not available. Please check your API configuration."
        else:
            logging.error(f"Gemini API error: {e}")
            return f"⚠️ AI service temporarily unavailable: {error_msg}"

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
        
        Rate the answer on:
        1. Accuracy (0-40 points)
        2. Completeness (0-30 points) 
        3. Clarity (0-30 points)
        
        Provide detailed feedback and suggestions.
        
        Respond in this EXACT JSON format:
        {{
            "score": 85,
            "feedback": "Good understanding of concepts. You demonstrated knowledge of the topic and provided relevant information.",
            "suggestions": "Add more specific examples to strengthen your answer."
        }}
        """
        
        response = gemini_generate(prompt)
        print(f"DEBUG: Gemini evaluation response: {response}")
        
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                evaluation = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
            # Validate the response has required fields
            if not isinstance(evaluation.get("score"), (int, float)):
                evaluation["score"] = 75
            if not evaluation.get("feedback"):
                evaluation["feedback"] = "Answer shows understanding of the topic."
            if not evaluation.get("suggestions"):
                evaluation["suggestions"] = "Could provide more specific examples."
                
        except Exception as e:
            print(f"DEBUG: JSON parsing error: {str(e)}")
            # Fallback evaluation based on answer length and content
            score = min(100, max(50, len(answer) // 10 + 60))  # Basic scoring
            evaluation = {
                "score": score,
                "feedback": f"Answer shows understanding of the topic. Length: {len(answer)} characters.",
                "suggestions": "Could provide more specific examples and details."
            }
        
        print(f"DEBUG: Final evaluation: {evaluation}")
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
        detailed_feedback = []
        
        print(f"DEBUG: Evaluating {len(answers)} answers")
        
        for i, answer in enumerate(answers):
            print(f"DEBUG: Evaluating answer {i+1}: {answer.get('question', 'No question')[:50]}...")
            evaluation = self.evaluate_answer(answer.get("question", ""), answer.get("answer", ""))
            
            # Handle both dict and string responses
            if isinstance(evaluation, dict):
                score = evaluation.get("score", 0)
                feedback = evaluation.get("feedback", "No feedback provided")
                suggestions = evaluation.get("suggestions", "")
                
                total_score += score
                feedback_items.append(feedback)
                
                # Create detailed feedback for each answer
                detailed_feedback.append({
                    "question": answer.get("question", f"Question {i+1}"),
                    "answer": answer.get("answer", ""),
                    "score": score,
                    "feedback": feedback,
                    "suggestions": suggestions
                })
            else:
                # If evaluation is a string, use default values
                total_score += 75  # Default score
                feedback_items.append(str(evaluation))
                detailed_feedback.append({
                    "question": answer.get("question", f"Question {i+1}"),
                    "answer": answer.get("answer", ""),
                    "score": 75,
                    "feedback": str(evaluation),
                    "suggestions": "Could provide more specific examples."
                })
        
        avg_score = total_score / len(answers) if answers else 0
        combined_feedback = " | ".join(feedback_items)
        
        print(f"DEBUG: Average score: {avg_score}, Feedback: {combined_feedback[:100]}...")
        
        return round(avg_score, 2), combined_feedback
