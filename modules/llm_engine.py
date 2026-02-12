"""
LLM Engine Module
Handles dual LLM architecture for question generation
"""

import os
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import json

try:
    from groq import Groq
    from langchain_groq import ChatGroq
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install groq langchain langchain-groq")


class DualLLMEngine:
    """
    Dual LLM Engine for question paper generation
    Uses two models: Parser (8B) and Generator (70B)
    """
    
    def __init__(
        self,
        parser_model: str = "llama-3.1-8b-instant",
        generator_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize Dual LLM Engine
        
        Args:
            parser_model: Model for parsing and evaluation
            generator_model: Model for question generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.parser_model = parser_model
        self.generator_model = generator_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Groq client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        
        # Initialize LangChain models
        self.parser_llm = ChatGroq(
            groq_api_key=api_key,
            model_name=parser_model,
            temperature=0.3,  # Lower temperature for parsing
            max_tokens=1024
        )
        
        self.generator_llm = ChatGroq(
            groq_api_key=api_key,
            model_name=generator_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def parse_syllabus(self, syllabus_text: str) -> Dict[str, Any]:
        """
        Parse syllabus to extract topics, units, and structure
        
        Args:
            syllabus_text: Raw syllabus text
        
        Returns:
            Structured syllabus data
        """
        prompt = f"""You are a syllabus parser. Extract and structure the following syllabus into JSON format.

Extract:
1. Course name
2. Units/Modules with their topics
3. Learning objectives
4. Suggested weightage (if mentioned)

Syllabus:
{syllabus_text}

Return ONLY a valid JSON object with this structure:
{{
    "course_name": "string",
    "units": [
        {{
            "unit_number": "string",
            "unit_name": "string",
            "topics": ["topic1", "topic2"],
            "weightage": 25
        }}
    ]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert syllabus analyzer. Always respond with valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.parser_llm.invoke(messages)
            content = response.content
            
            # Extract JSON from response
            json_str = self._extract_json(content)
            parsed_data = json.loads(json_str)
            
            return parsed_data
        except Exception as e:
            print(f"Error parsing syllabus: {e}")
            return {"course_name": "Unknown", "units": []}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_questions(
        self,
        syllabus: str,
        topics: List[Dict[str, Any]],
        bloom_distribution: Dict[str, int],
        total_marks: int
    ) -> List[Dict[str, Any]]:
        """
        Generate questions based on syllabus and parameters
        
        Args:
            syllabus: Syllabus text
            topics: List of topics with priorities
            bloom_distribution: Bloom's taxonomy distribution
            total_marks: Total marks for the paper
        
        Returns:
            List of generated questions
        """
        # Calculate number of questions per Bloom's level
        questions_per_level = self._calculate_questions_per_level(
            bloom_distribution, total_marks
        )
        
        all_questions = []
        question_number = 1
        
        # Generate questions for each Bloom's level
        for level, count_data in questions_per_level.items():
            if count_data['count'] == 0:
                continue
            
            level_questions = self._generate_questions_for_level(
                level=level,
                count=count_data['count'],
                marks_per_question=count_data['marks_per_question'],
                topics=topics,
                syllabus=syllabus,
                start_number=question_number
            )
            
            all_questions.extend(level_questions)
            question_number += len(level_questions)
        
        return all_questions
    
    def _calculate_questions_per_level(
        self, bloom_distribution: Dict[str, int], total_marks: int
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate how many questions to generate for each Bloom's level"""
        questions_distribution = {}
        
        # Typical marks per question for each level
        marks_mapping = {
            'remember': 2,
            'understand': 3,
            'apply': 5,
            'analyze': 8,
            'evaluate': 10,
            'create': 15
        }
        
        for level, percentage in bloom_distribution.items():
            level_lower = level.lower()
            marks_allocated = int((percentage / 100) * total_marks)
            marks_per_q = marks_mapping.get(level_lower, 5)
            question_count = max(1, marks_allocated // marks_per_q)
            
            questions_distribution[level] = {
                'count': question_count,
                'marks_per_question': marks_per_q,
                'total_marks': marks_allocated
            }
        
        return questions_distribution
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_questions_for_level(
        self,
        level: str,
        count: int,
        marks_per_question: int,
        topics: List[Dict[str, Any]],
        syllabus: str,
        start_number: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate questions for a specific Bloom's level"""
        
        # Select top topics by priority
        selected_topics = sorted(topics, key=lambda x: x.get('priority', 0), reverse=True)[:5]
        topics_str = ", ".join([t.get('name', t.get('topic', 'Unknown')) for t in selected_topics])
        
        # Create prompt for question generation
        prompt = f"""Generate {count} examination questions at the "{level}" level of Bloom's Taxonomy.

Each question should be worth {marks_per_question} marks.

Topics to cover: {topics_str}

Syllabus context:
{syllabus[:1000]}...

Requirements:
1. Questions must be at the "{level}" cognitive level
2. Cover different topics
3. Be clear, unambiguous, and exam-appropriate
4. Include enough context for students to answer

Bloom's Level Description:
- Remember: Recall facts, terms, basic concepts
- Understand: Explain ideas or concepts
- Apply: Use knowledge in new situations
- Analyze: Break down into parts, find relationships
- Evaluate: Justify decisions, critique
- Create: Design, construct, produce something new

Generate {count} questions in JSON format:
[
    {{
        "question": "Question text here",
        "marks": {marks_per_question},
        "bloom_level": "{level}",
        "topic": "Specific topic",
        "difficulty": "easy/medium/hard"
    }}
]

Return ONLY valid JSON array."""
        
        messages = [
            SystemMessage(content="You are an expert question paper setter with deep knowledge of Bloom's Taxonomy."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.generator_llm.invoke(messages)
            content = response.content
            
            # Extract JSON from response
            json_str = self._extract_json(content)
            questions = json.loads(json_str)
            
            # Add question numbers
            for i, q in enumerate(questions):
                q['number'] = start_number + i
            
            return questions
        except Exception as e:
            print(f"Error generating questions for {level}: {e}")
            # Return fallback questions
            return self._generate_fallback_questions(level, count, marks_per_question, start_number)
    
    def _generate_fallback_questions(
        self, level: str, count: int, marks: int, start_number: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback questions if API fails"""
        fallback = []
        for i in range(count):
            fallback.append({
                'number': start_number + i,
                'question': f"[Placeholder {level} question {i+1}]",
                'marks': marks,
                'bloom_level': level,
                'topic': 'General',
                'difficulty': 'medium'
            })
        return fallback
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response"""
        # Try to find JSON in code blocks
        import re
        
        # Look for JSON in code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Look for JSON object/array directly
        json_pattern = r'(\{.*?\}|\[.*?\])'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        return text
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def evaluate_question_quality(self, question: Dict[str, Any]) -> float:
        """
        Evaluate quality of a generated question
        
        Args:
            question: Question dictionary
        
        Returns:
            Quality score (0-10)
        """
        prompt = f"""Evaluate the quality of this examination question on a scale of 0-10.

Question: {question.get('question', '')}
Marks: {question.get('marks', 0)}
Bloom's Level: {question.get('bloom_level', '')}
Topic: {question.get('topic', '')}

Evaluation criteria:
1. Clarity and unambiguity (0-3)
2. Appropriate for Bloom's level (0-3)
3. Exam suitability (0-2)
4. Scope matches marks (0-2)

Return ONLY a number between 0 and 10."""
        
        messages = [
            SystemMessage(content="You are an exam quality evaluator."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.parser_llm.invoke(messages)
            score_str = response.content.strip()
            
            # Extract number from response
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', score_str)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))
            
            return 7.0  # Default score
        except Exception as e:
            print(f"Error evaluating question: {e}")
            return 7.0
    
    def refine_question(self, question: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        """
        Refine a question based on feedback
        
        Args:
            question: Original question
            feedback: Feedback for improvement
        
        Returns:
            Refined question
        """
        prompt = f"""Improve this examination question based on the feedback.

Original Question: {question.get('question', '')}
Marks: {question.get('marks', 0)}
Bloom's Level: {question.get('bloom_level', '')}

Feedback: {feedback}

Provide an improved version of the question that addresses the feedback.
Return ONLY the improved question text, nothing else."""
        
        try:
            response = self.generator_llm.invoke([HumanMessage(content=prompt)])
            improved_question = question.copy()
            improved_question['question'] = response.content.strip()
            return improved_question
        except Exception as e:
            print(f"Error refining question: {e}")
            return question
