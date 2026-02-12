"""
ChromaDB Handler Module
Handles vector database operations for storing and retrieving syllabus and question data
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install chromadb sentence-transformers")


class ChromaDBHandler:
    """
    Handler for ChromaDB vector database operations
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "question_papers"
    ):
        """
        Initialize ChromaDB Handler
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
        """
        # Set persist directory
        if persist_directory is None:
            persist_directory = os.getenv('CHROMADB_PATH', './data/chromadb')
        
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(persist_path)
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Syllabus and question paper knowledge base"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.collection_name = collection_name
    
    def store_syllabus(
        self,
        syllabus_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store syllabus in vector database
        
        Args:
            syllabus_text: Syllabus text content
            metadata: Additional metadata
        
        Returns:
            Document ID
        """
        # Generate embedding
        embedding = self.embedding_model.encode([syllabus_text])[0].tolist()
        
        # Create unique ID
        doc_id = f"syllabus_{metadata.get('filename', 'unknown')}_{hash(syllabus_text) % 10000}"
        
        # Prepare metadata
        meta = metadata or {}
        meta['type'] = 'syllabus'
        meta['content_length'] = len(syllabus_text)
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[syllabus_text],
            metadatas=[meta],
            ids=[doc_id]
        )
        
        return doc_id
    
    def store_question(
        self,
        question: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a question in vector database
        
        Args:
            question: Question dictionary
            metadata: Additional metadata
        
        Returns:
            Document ID
        """
        question_text = question.get('question', '')
        
        if not question_text:
            raise ValueError("Question text is empty")
        
        # Generate embedding
        embedding = self.embedding_model.encode([question_text])[0].tolist()
        
        # Create unique ID
        q_num = question.get('number', 0)
        doc_id = f"question_{q_num}_{hash(question_text) % 10000}"
        
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            'type': 'question',
            'marks': question.get('marks', 0),
            'bloom_level': question.get('bloom_level', ''),
            'topic': question.get('topic', ''),
            'difficulty': question.get('difficulty', 'medium')
        })
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[question_text],
            metadatas=[meta],
            ids=[doc_id]
        )
        
        return doc_id
    
    def store_questions_batch(
        self,
        questions: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Store multiple questions in batch
        
        Args:
            questions: List of question dictionaries
            metadata: Common metadata for all questions
        
        Returns:
            List of document IDs
        """
        if not questions:
            return []
        
        # Extract question texts
        question_texts = [q.get('question', '') for q in questions]
        question_texts = [qt for qt in question_texts if qt]  # Filter empty
        
        if not question_texts:
            return []
        
        # Generate embeddings in batch
        embeddings = self.embedding_model.encode(question_texts).tolist()
        
        # Prepare data
        ids = []
        metadatas = []
        
        for i, q in enumerate(questions):
            q_text = q.get('question', '')
            if not q_text:
                continue
            
            q_num = q.get('number', i)
            doc_id = f"question_{q_num}_{hash(q_text) % 10000}"
            ids.append(doc_id)
            
            meta = metadata.copy() if metadata else {}
            meta.update({
                'type': 'question',
                'marks': q.get('marks', 0),
                'bloom_level': q.get('bloom_level', ''),
                'topic': q.get('topic', ''),
                'difficulty': q.get('difficulty', 'medium')
            })
            metadatas.append(meta)
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=question_texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search_similar(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Filter by metadata
        
        Returns:
            Search results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        return results
    
    def get_syllabus_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Get relevant syllabus context for a query
        
        Args:
            query: Query string
            top_k: Number of results
        
        Returns:
            List of relevant syllabus chunks
        """
        results = self.search_similar(
            query,
            n_results=top_k,
            filter_metadata={"type": "syllabus"}
        )
        
        # Extract documents
        documents = results.get('documents', [[]])[0]
        return documents
    
    def find_similar_questions(
        self,
        question_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find similar questions to avoid duplicates
        
        Args:
            question_text: Question to check
            top_k: Number of results
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of similar questions
        """
        results = self.search_similar(
            question_text,
            n_results=top_k,
            filter_metadata={"type": "question"}
        )
        
        # Format results
        similar_questions = []
        
        if results.get('documents'):
            documents = results['documents'][0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1 - dist
                
                if similarity >= similarity_threshold:
                    similar_questions.append({
                        'question': doc,
                        'metadata': meta,
                        'similarity': similarity
                    })
        
        return similar_questions
    
    def get_all_questions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all stored questions
        
        Args:
            limit: Maximum number of questions to return
        
        Returns:
            List of questions
        """
        # Query with filter
        results = self.collection.get(
            where={"type": "question"},
            limit=limit
        )
        
        questions = []
        
        if results.get('documents'):
            documents = results['documents']
            metadatas = results.get('metadatas', [])
            
            for doc, meta in zip(documents, metadatas):
                questions.append({
                    'question': doc,
                    'marks': meta.get('marks', 0),
                    'bloom_level': meta.get('bloom_level', ''),
                    'topic': meta.get('topic', ''),
                    'difficulty': meta.get('difficulty', 'medium')
                })
        
        return questions
    
    def delete_document(self, doc_id: str):
        """Delete a document by ID"""
        self.collection.delete(ids=[doc_id])
    
    def clear_collection(self):
        """Clear all documents from collection"""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Syllabus and question paper knowledge base"}
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the database
        
        Returns:
            Statistics dictionary
        """
        # Get total count
        all_docs = self.collection.get()
        total_count = len(all_docs.get('ids', []))
        
        # Count by type
        syllabi_count = 0
        questions_count = 0
        
        for meta in all_docs.get('metadatas', []):
            doc_type = meta.get('type', '')
            if doc_type == 'syllabus':
                syllabi_count += 1
            elif doc_type == 'question':
                questions_count += 1
        
        return {
            'total_documents': total_count,
            'syllabi_count': syllabi_count,
            'questions_count': questions_count,
            'collection_name': self.collection_name
        }
