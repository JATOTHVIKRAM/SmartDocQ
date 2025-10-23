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

        # Check if this is MCQ type
        if qtype and qtype.lower() == "mcq":
            return self._generate_mcq_questions(num_questions, level, context)
        else:
            return self._generate_text_questions(num_questions, level, qtype, context)

    def _generate_mcq_questions(self, num_questions: int, level: str, context: str) -> List[str]:
        """Generate MCQ questions with options"""
        prompt = f"""
        Based on the following document content, generate {num_questions} Multiple Choice Questions (MCQ).
        
        Question Level: {level}
        
        Document Content:
        {context}
        
        For each question, provide:
        1. A clear question
        2. Four options (A, B, C, D)
        3. The correct answer
        
        Respond in this EXACT JSON format:
        [
            {{
                "question": "What is the main topic discussed in the document?",
                "options": [
                    "A) Option 1",
                    "B) Option 2", 
                    "C) Option 3",
                    "D) Option 4"
                ],
                "correct": "A) Option 1"
            }}
        ]
        
        IMPORTANT: 
        - Each question must have exactly 4 options (A, B, C, D)
        - The correct answer must match one of the options exactly
        - Make sure options are relevant to the document content
        
        Generate {num_questions} relevant MCQ questions that test understanding of this content.
        """
        
        response = gemini_generate(prompt)
        print(f"DEBUG: MCQ generation response: {response}")
        
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                structured_questions = json.loads(json_str)
                
                # Validate and clean up the questions
                validated_questions = []
                for item in structured_questions:
                    if isinstance(item, dict) and item.get("question") and item.get("options"):
                        options = [str(opt).strip() for opt in item["options"] if str(opt).strip()]
                        if len(options) == 4:  # Ensure exactly 4 options
                            validated_questions.append({
                                "question": str(item["question"]).strip(),
                                "options": options,
                                "correct": str(item.get("correct", "")).strip()
                            })
                
                if len(validated_questions) >= num_questions:
                    self.structured_questions = validated_questions[:num_questions]
                    self.questions = [q["question"] for q in self.structured_questions]
                    print(f"DEBUG: Generated {len(self.structured_questions)} MCQ questions with options")
                    return self.questions
        except Exception as e:
            print(f"DEBUG: MCQ JSON parsing failed: {str(e)}")
        
        # Fallback to text questions if MCQ generation fails
        print("DEBUG: Falling back to text questions")
        return self._generate_text_questions(num_questions, level, "general", context)

    def _generate_text_questions(self, num_questions: int, level: str, qtype: Optional[str], context: str) -> List[str]:
        """Generate regular text questions with model answers for technical questions"""
        if qtype and qtype.lower() == "technical":
            return self._generate_technical_questions(num_questions, level, context)
        else:
            return self._generate_general_questions(num_questions, level, qtype, context)

    def _generate_technical_questions(self, num_questions: int, level: str, context: str) -> List[str]:
        """Generate technical questions with model answers, IDs, and difficulty levels"""
        prompt = f"""
        Based on the following document content, generate {num_questions} technical interview questions.
        
        Question Level: {level}
        
        Document Content:
        {context}
        
        For each question, provide:
        1. A unique question ID
        2. A clear technical question
        3. A difficulty level (Easy, Medium, Hard)
        4. A comprehensive model/correct answer
        5. Key concepts that should be covered
        
        Respond in this EXACT JSON format:
        [
            {{
                "id": "tech_001",
                "question": "Explain the main concept discussed in the document",
                "difficulty": "Medium",
                "model_answer": "The main concept is... [detailed explanation with key technical details]",
                "key_concepts": ["concept1", "concept2", "concept3"]
            }}
        ]
        
        IMPORTANT:
        - Generate questions that test deep technical understanding
        - Make model answers comprehensive and detailed
        - Ensure difficulty levels match the requested level ({level})
        - Include specific technical terms and concepts from the document
        
        Generate {num_questions} relevant technical questions that test understanding of this content.
        """
        
        response = gemini_generate(prompt)
        print(f"DEBUG: Technical questions generation response: {response}")
        
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                structured_questions = json.loads(json_str)
                
                # Validate and clean up the questions
                validated_questions = []
                for i, item in enumerate(structured_questions):
                    if isinstance(item, dict) and item.get("question"):
                        validated_questions.append({
                            "id": item.get("id", f"tech_{i+1:03d}"),
                            "question": str(item["question"]).strip(),
                            "difficulty": item.get("difficulty", level.title()),
                            "model_answer": str(item.get("model_answer", "")).strip(),
                            "key_concepts": [str(concept).strip() for concept in item.get("key_concepts", []) if str(concept).strip()]
                        })
                
                if len(validated_questions) >= num_questions:
                    self.structured_questions = validated_questions[:num_questions]
                    self.questions = [q["question"] for q in self.structured_questions]
                    print(f"DEBUG: Generated {len(self.structured_questions)} technical questions with metadata")
                    return self.questions
        except Exception as e:
            print(f"DEBUG: Technical questions JSON parsing failed: {str(e)}")
        
        # Fallback to general questions if technical generation fails
        print("DEBUG: Falling back to general questions")
        return self._generate_general_questions(num_questions, level, "general", context)

    def _generate_general_questions(self, num_questions: int, level: str, qtype: Optional[str], context: str) -> List[str]:
        """Generate general text questions"""
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
        """Evaluate user's answer and provide feedback using Gemini AI."""
        if not self.chunks:
            return {"score": 0, "feedback": "No document loaded for evaluation."}

        # Get relevant context
        context = "\n".join(self.chunks[:3])

        # Check if this is a technical question with model answer
        model_answer = None
        key_concepts = []
        question_id = None
        difficulty = None
        if hasattr(self, 'structured_questions') and self.structured_questions:
            for sq in self.structured_questions:
                if sq.get("question") == question:
                    model_answer = sq.get("model_answer", "")
                    key_concepts = sq.get("key_concepts", [])
                    question_id = sq.get("id", "")
                    difficulty = sq.get("difficulty", "")
                    break

        if model_answer and key_concepts:
            # Use semantic similarity evaluation for technical questions
            similarity = self._calculate_semantic_similarity(answer, model_answer, key_concepts)
            points, verdict = self._convert_similarity_to_score(similarity)
            
            # Generate AI feedback based on similarity
            if similarity >= 0.85:
                feedback_text = f"Excellent answer! You demonstrated strong understanding of the key concepts."
                suggestions = "Continue building on this knowledge with more advanced topics."
            elif similarity >= 0.60:
                feedback_text = f"Good answer with some key points covered. You understood {int(similarity*100)}% of the expected content."
                suggestions = "Try to include more technical details and specific examples from the document."
            else:
                feedback_text = f"Your answer needs improvement. You covered {int(similarity*100)}% of the expected content."
                suggestions = "Review the document content and focus on the key technical concepts mentioned."
            
            return {
                "score": int(points * 100),  # Convert to 0-100 scale for display
                "points": points,  # Keep original 0-1 scale
                "feedback": feedback_text,
                "suggestions": suggestions,
                "status": verdict,
                "model_answer": model_answer,
                "similarity": similarity,
                "question_id": question_id,
                "difficulty": difficulty
            }
        else:
        # Standard evaluation for general questions
        prompt = f"""
        You are an expert interviewer evaluating a candidate's answer. Rate this answer based on the document content.
        
        Question: {question}
        Candidate's Answer: {answer}
        
        Document Context (for reference):
        {context}
        
        Evaluation Criteria:
        1. RELEVANCE (0-30 points): How well does the answer address the question?
        2. ACCURACY (0-30 points): How accurate is the information based on the document?
        3. DEPTH (0-25 points): How detailed and comprehensive is the answer?
        4. CLARITY (0-15 points): How clear and well-structured is the response?
        
        IMPORTANT: You MUST respond with ONLY valid JSON. No additional text or explanations.
        
        Respond in this EXACT JSON format:
        {{
            "score": 85,
            "feedback": "Your answer demonstrates good understanding of the topic. You correctly identified the key concepts and provided relevant examples. The explanation was clear and well-structured.",
            "suggestions": "To improve further, consider adding more specific details from the document and providing concrete examples to support your points.",
            "status": "Partially Correct"
        }}
        
        Status should be one of: "Correct", "Partially Correct", or "Incorrect"
        """
        
        print(f"DEBUG: Evaluating question: {question[:50]}...")
        print(f"DEBUG: Evaluating answer: {answer[:50]}...")
        
        response = gemini_generate(prompt)
        print(f"DEBUG: Gemini evaluation response type: {type(response)}")
        print(f"DEBUG: Gemini evaluation response: {response}")
        
        # Check if response is an error string
        if isinstance(response, str) and (response.startswith("Error") or response.startswith("⚠️")):
            print(f"DEBUG: Gemini returned error: {response}")
            score = self._fallback_score_evaluation(question, answer, context)
            return {
                "score": score,
                "feedback": f"Answer evaluated using content analysis. Score based on relevance, structure, and detail.",
                "suggestions": "Provide more specific examples and detailed explanations to improve your score.",
                "status": "Partially Correct",
                "model_answer": model_answer or "No model answer available"
            }
        
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                evaluation = json.loads(json_str)
                
                # Validate the response has required fields
                if not isinstance(evaluation.get("score"), (int, float)):
                    evaluation["score"] = self._fallback_score_evaluation(question, answer, context)
                if not evaluation.get("feedback"):
                    evaluation["feedback"] = "Answer shows understanding of the topic."
                if not evaluation.get("suggestions"):
                    evaluation["suggestions"] = "Could provide more specific examples."
                if not evaluation.get("status"):
                    evaluation["status"] = "Partially Correct"
                
                # Add model answer if available
                if model_answer:
                    evaluation["model_answer"] = model_answer
                
                print(f"DEBUG: Successfully parsed Gemini evaluation: {evaluation}")
                return evaluation
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"DEBUG: JSON parsing error: {str(e)}")
            # Fallback evaluation based on answer content analysis
            score = self._fallback_score_evaluation(question, answer, context)
            evaluation = {
                "score": score,
                "feedback": f"Answer evaluated using content analysis. Score based on relevance, structure, and detail.",
                "suggestions": "Provide more specific examples and detailed explanations to improve your score.",
                "status": "Partially Correct",
                "model_answer": model_answer or "No model answer available"
            }
        
        print(f"DEBUG: Final evaluation: {evaluation}")
        return evaluation

    def _calculate_semantic_similarity(self, user_answer: str, model_answer: str, key_concepts: List[str]) -> float:
        """Calculate semantic similarity between user answer and model answer"""
        if not user_answer or not model_answer:
            return 0.0
        
        user_lower = user_answer.lower()
        model_lower = model_answer.lower()
        
        # Keyword matching for key concepts
        concept_score = 0.0
        if key_concepts:
            matched_concepts = 0
            for concept in key_concepts:
                concept_lower = concept.lower()
                if concept_lower in user_lower:
                    matched_concepts += 1
            concept_score = matched_concepts / len(key_concepts)
        
        # Text similarity using TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([user_answer, model_answer])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Combine concept matching and text similarity
            # Weight: 60% text similarity, 40% concept matching
            final_score = (0.6 * similarity) + (0.4 * concept_score)
            
            return min(1.0, max(0.0, final_score))
        except Exception as e:
            print(f"DEBUG: TF-IDF similarity calculation failed: {str(e)}")
            # Fallback to simple keyword matching
            return concept_score

    def _convert_similarity_to_score(self, similarity: float) -> tuple:
        """Convert similarity score to points and verdict"""
        if similarity >= 0.85:
            return 1.0, "Correct"
        elif similarity >= 0.60:
            return 0.5, "Partially Correct"
        else:
            return 0.0, "Incorrect"

    def _fallback_score_evaluation(self, question: str, answer: str, context: str) -> int:
        """Fallback scoring when Gemini JSON parsing fails - actually evaluates content"""
        if not answer or len(answer.strip()) < 5:
            return 15  # Very short or empty answer
        
        score = 0
        answer_lower = answer.lower().strip()
        
        # Check for completely wrong/uncertain answers first
        wrong_indicators = ['i don\'t know', 'no idea', 'not sure', 'maybe', 'i think', 'probably', 'wrong', 'incorrect', 'i have no idea', 'i\'m not sure', 'i don\'t understand', 'idk', 'dunno']
        if any(indicator in answer_lower for indicator in wrong_indicators):
            return 20  # Very low score for uncertain answers
        
        # Check for explicitly wrong answers
        if 'wrong' in answer_lower or 'incorrect' in answer_lower or 'false' in answer_lower or 'no' in answer_lower:
            return 25  # Low score for wrong answers
        
        # Length scoring (0-20 points)
        if len(answer) >= 100:
            score += 20
        elif len(answer) >= 50:
            score += 15
        elif len(answer) >= 20:
            score += 10
        else:
            score += 5
        
        # Content analysis (0-40 points)
        question_lower = question.lower()
        
        # Check if answer addresses the question
        question_words = set(question_lower.split())
        answer_words = set(answer_lower.split())
        common_words = question_words.intersection(answer_words)
        
        if len(common_words) > 0:
            score += 25  # Answer relates to question
        else:
            score += 5   # Answer doesn't relate to question
        
        # Check for good answer indicators
        good_indicators = ['because', 'therefore', 'example', 'specifically', 'detail', 'explain', 'demonstrate', 'according to', 'based on']
        if any(indicator in answer_lower for indicator in good_indicators):
            score += 20  # Bonus for detailed explanations
        
        # Context relevance (0-20 points)
        if context:
            context_words = set(context.lower().split())
            context_overlap = len(answer_words.intersection(context_words))
            if context_overlap > 5:
                score += 20  # Good context usage
            elif context_overlap > 2:
                score += 10  # Some context usage
            else:
                score += 5   # Little context usage
        
        # Structure scoring (0-20 points)
        if '.' in answer and len(answer.split('.')) > 2:
            score += 10  # Multiple sentences
        if ',' in answer:
            score += 5   # Uses commas (better structure)
        if answer[0].isupper() and answer.endswith('.'):
            score += 5   # Proper sentence structure
        
        # Ensure score is between 0-100
        return max(0, min(100, score))

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
            print(f"DEBUG: Evaluation result type: {type(evaluation)}")
            print(f"DEBUG: Evaluation result: {evaluation}")
            
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
        
        # Return detailed feedback for each answer instead of combined text
        return avg_score, detailed_feedback
