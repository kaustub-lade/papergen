"""
Novelty Filter Module
Filters duplicate and similar questions using semantic similarity
"""

from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install sentence-transformers chromadb")


class NoveltyFilter:
    """
    Filter duplicate and similar questions using cosine similarity
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize Novelty Filter
        
        Args:
            similarity_threshold: Threshold for considering questions as duplicates (0-1)
            embedding_model: Sentence transformer model name
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model
        
        # Load embedding model
        try:
            self.model = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.model = None
        
        # Store question embeddings for comparison
        self.question_embeddings = []
        self.questions_seen = []
    
    def filter(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out duplicate and similar questions
        
        Args:
            questions: List of question dictionaries
        
        Returns:
            Filtered list of unique questions
        """
        if not self.model:
            print("Warning: Embedding model not loaded. Returning all questions.")
            return questions
        
        unique_questions = []
        
        for question in questions:
            question_text = question.get('question', '')
            
            if not question_text:
                continue
            
            # Check if question is novel
            if self.is_novel(question_text):
                unique_questions.append(question)
                # Add to seen questions
                self.add_question(question_text)
        
        return unique_questions
    
    def is_novel(self, question_text: str, threshold: Optional[float] = None) -> bool:
        """
        Check if a question is novel compared to previously seen questions
        
        Args:
            question_text: Question to check
            threshold: Custom similarity threshold (uses default if None)
        
        Returns:
            True if novel, False if similar to existing question
        """
        if not self.model or len(self.questions_seen) == 0:
            return True
        
        threshold = threshold or self.similarity_threshold
        
        # Get embedding for new question
        new_embedding = self.model.encode([question_text])[0]
        
        # Calculate cosine similarity with all existing questions
        for existing_embedding in self.question_embeddings:
            similarity = self._cosine_similarity(new_embedding, existing_embedding)
            
            if similarity >= threshold:
                # Too similar to an existing question
                return False
        
        return True
    
    def add_question(self, question_text: str):
        """
        Add a question to the seen questions list
        
        Args:
            question_text: Question to add
        """
        if not self.model:
            return
        
        embedding = self.model.encode([question_text])[0]
        self.question_embeddings.append(embedding)
        self.questions_seen.append(question_text)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))
    
    def find_similar_questions(
        self,
        question_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top K most similar questions
        
        Args:
            question_text: Query question
            top_k: Number of similar questions to return
        
        Returns:
            List of similar questions with similarity scores
        """
        if not self.model or len(self.questions_seen) == 0:
            return []
        
        # Get embedding for query question
        query_embedding = self.model.encode([question_text])[0]
        
        # Calculate similarities
        similarities = []
        for i, existing_embedding in enumerate(self.question_embeddings):
            similarity = self._cosine_similarity(query_embedding, existing_embedding)
            similarities.append({
                'question': self.questions_seen[i],
                'similarity': similarity
            })
        
        # Sort by similarity and return top K
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def clear(self):
        """Clear all seen questions"""
        self.question_embeddings = []
        self.questions_seen = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about filtered questions
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_questions_seen': len(self.questions_seen),
            'similarity_threshold': self.similarity_threshold,
            'embedding_model': self.embedding_model_name
        }
    
    def batch_filter_with_past_papers(
        self,
        questions: List[Dict[str, Any]],
        past_papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter questions against past papers
        
        Args:
            questions: New questions to filter
            past_papers: Past question papers
        
        Returns:
            Filtered questions
        """
        # First, add all past paper questions to seen list
        for paper in past_papers:
            paper_questions = paper.get('questions', [])
            for q in paper_questions:
                q_text = q.get('question', '') if isinstance(q, dict) else str(q)
                if q_text:
                    self.add_question(q_text)
        
        # Now filter new questions
        return self.filter(questions)
    
    def diversity_score(self, questions: List[Dict[str, Any]]) -> float:
        """
        Calculate diversity score for a set of questions
        
        Args:
            questions: List of questions
        
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if not self.model or len(questions) < 2:
            return 1.0
        
        # Get embeddings for all questions
        question_texts = [q.get('question', '') for q in questions]
        embeddings = self.model.encode(question_texts)
        
        # Calculate pairwise similarities
        similarities = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, diversity))
