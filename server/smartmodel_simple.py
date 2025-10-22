# smartmodel.py - Simplified version for Render deployment
import os
from typing import List, Optional
import docx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in environment")
genai.configure(api_key=API_KEY)

# ---------------------
# Document reading
# ---------------------
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

# ---------------------
# Text processing
# ---------------------
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

# ---------------------
# Simple similarity search using TF-IDF
# ---------------------
class SimpleDocumentQA:
    def __init__(self):
        self.chunks: Optional[List[str]] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None

    def load_document(self, file_path: str) -> str:
        """Load and index document using TF-IDF"""
        text = read_document_file(file_path)
        if not text or not text.strip():
            raise ValueError("Document empty after extraction")
        
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        if not chunks:
            raise ValueError("No chunks created from document")
        
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        
        return f"Document loaded. {len(self.chunks)} chunks indexed."

    def ask_question(self, query: str) -> str:
        """Answer question using TF-IDF similarity"""
        if not query or not query.strip():
            return "Please ask a valid question."
        
        if self.chunks is None or self.vectorizer is None or self.tfidf_matrix is None:
            return "⚠️ Please load a document first."
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top 3 most similar chunks
        top_indices = similarities.argsort()[-3:][::-1]
        top_chunks = [self.chunks[i] for i in top_indices if similarities[i] > 0.1]
        
        if not top_chunks:
            return "Sorry — I cannot find the answer in the document."
        
        # Send to Gemini with context
        return ask_gemini_with_context(query, top_chunks)

    def clear_cache(self) -> str:
        """Clear loaded document"""
        self.chunks = None
        self.vectorizer = None
        self.tfidf_matrix = None
        return "Cache cleared"

def ask_gemini_with_context(query: str, context_chunks: List[str]) -> str:
    """Ask Gemini with document context"""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        context_text = "\n".join(context_chunks)
        prompt = f"""
You are a document Q&A assistant. Use the context below to answer the question.
If the answer is clear, respond directly. If partial, say: 'Based on the document, [summarize]. For details, ask about a specific section.'

Context:
{context_text}

Question:
{query}

Answer:
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ---------------------
# Public API - DocumentQA class
# ---------------------
class DocumentQA:
    def __init__(self):
        self._qa = SimpleDocumentQA()

    def load_document(self, file_path: str) -> str:
        return self._qa.load_document(file_path)

    def ask_question(self, query: str) -> str:
        return self._qa.ask_question(query)

    def clear_cache(self) -> str:
        return self._qa.clear_cache()
