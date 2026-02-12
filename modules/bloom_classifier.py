"""
Bloom's Taxonomy Classifier Module
Classifies questions into Bloom's taxonomy cognitive levels
"""

import re
from typing import Dict, List, Any
import numpy as np


class BloomClassifier:
    """
    Classifier for Bloom's Taxonomy cognitive levels
    """
    
    # Keywords associated with each Bloom's level
    BLOOM_KEYWORDS = {
        'remember': [
            'define', 'list', 'recall', 'identify', 'name', 'state', 'describe',
            'recognize', 'select', 'label', 'match', 'memorize', 'what', 'when',
            'where', 'who', 'choose', 'find', 'show', 'spell', 'tell', 'write'
        ],
        'understand': [
            'explain', 'summarize', 'paraphrase', 'describe', 'interpret',
            'discuss', 'distinguish', 'predict', 'associate', 'estimate',
            'differentiate', 'extend', 'give examples', 'infer', 'outline',
            'relate', 'rephrase', 'translate', 'elaborate', 'comprehend'
        ],
        'apply': [
            'apply', 'calculate', 'change', 'demonstrate', 'discover',
            'manipulate', 'modify', 'operate', 'predict', 'prepare',
            'produce', 'relate', 'show', 'solve', 'use', 'implement',
            'construct', 'develop', 'organize', 'utilize'
        ],
        'analyze': [
            'analyze', 'break down', 'compare', 'contrast', 'diagram',
            'deconstruct', 'differentiate', 'discriminate', 'distinguish',
            'identify', 'illustrate', 'infer', 'outline', 'relate',
            'select', 'separate', 'examine', 'categorize', 'classify',
            'investigate', 'deduce'
        ],
        'evaluate': [
            'appraise', 'argue', 'defend', 'judge', 'select', 'support',
            'value', 'critique', 'weigh', 'assess', 'decide', 'rate',
            'choose', 'recommend', 'justify', 'prioritize', 'determine',
            'evaluate', 'conclude', 'criticize'
        ],
        'create': [
            'design', 'construct', 'develop', 'create', 'formulate',
            'author', 'investigate', 'compose', 'generate', 'modify',
            'plan', 'produce', 'propose', 'assemble', 'devise', 'invent',
            'build', 'synthesize', 'integrate', 'combine'
        ]
    }
    
    def __init__(self, use_ml: bool = False):
        """
        Initialize Bloom's Taxonomy Classifier
        
        Args:
            use_ml: Whether to use ML model (if trained) for classification
        """
        self.use_ml = use_ml
        self.ml_model = None
        
        if use_ml:
            self._load_ml_model()
    
    def classify_question(self, question: str) -> str:
        """
        Classify a question into Bloom's taxonomy level
        
        Args:
            question: Question text
        
        Returns:
            Bloom's level ('remember', 'understand', 'apply', etc.)
        """
        if self.use_ml and self.ml_model is not None:
            return self._classify_with_ml(question)
        else:
            return self._classify_with_keywords(question)
    
    def _classify_with_keywords(self, question: str) -> str:
        """Classify using keyword matching"""
        question_lower = question.lower()
        
        # Score each level based on keyword presence
        level_scores = {}
        
        for level, keywords in self.BLOOM_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in question_lower:
                    # Give higher weight to keywords at the beginning
                    position = question_lower.find(keyword)
                    if position < len(question_lower) * 0.3:  # First 30% of question
                        score += 2
                    else:
                        score += 1
            
            level_scores[level] = score
        
        # Additional heuristics
        level_scores = self._apply_heuristics(question_lower, level_scores)
        
        # Return level with highest score
        if max(level_scores.values()) == 0:
            return 'understand'  # Default level
        
        return max(level_scores, key=level_scores.get)
    
    def _apply_heuristics(self, question: str, scores: Dict[str, int]) -> Dict[str, int]:
        """Apply additional heuristics for classification"""
        
        # Question word patterns
        if any(word in question for word in ['what', 'who', 'when', 'where']):
            scores['remember'] += 1
        
        if any(word in question for word in ['why', 'how', 'explain']):
            scores['understand'] += 1
        
        if any(word in question for word in ['calculate', 'compute', 'solve']):
            scores['apply'] += 2
        
        if any(word in question for word in ['compare', 'contrast', 'analyze']):
            scores['analyze'] += 2
        
        if any(word in question for word in ['evaluate', 'judge', 'assess', 'critique']):
            scores['evaluate'] += 2
        
        if any(word in question for word in ['design', 'create', 'develop', 'propose']):
            scores['create'] += 2
        
        # Complexity indicators
        if len(question.split()) > 30:  # Long questions tend to be higher-order
            scores['analyze'] += 1
            scores['evaluate'] += 1
            scores['create'] += 1
        
        # Multiple parts indicate higher cognitive load
        if question.count('?') > 1 or any(word in question for word in ['and', 'or', 'also']):
            scores['analyze'] += 1
        
        return scores
    
    def _classify_with_ml(self, question: str) -> str:
        """Classify using trained ML model"""
        # TODO: Implement ML-based classification
        # For now, fallback to keyword-based
        return self._classify_with_keywords(question)
    
    def _load_ml_model(self):
        """Load pre-trained ML model for classification"""
        # TODO: Implement model loading
        # from models/ directory
        pass
    
    def classify_batch(self, questions: List[str]) -> List[str]:
        """
        Classify multiple questions
        
        Args:
            questions: List of question texts
        
        Returns:
            List of Bloom's levels
        """
        return [self.classify_question(q) for q in questions]
    
    def get_level_distribution(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get distribution of Bloom's levels in a set of questions
        
        Args:
            questions: List of question dictionaries
        
        Returns:
            Dictionary with count of each level
        """
        distribution = {level: 0 for level in self.BLOOM_KEYWORDS.keys()}
        
        for question in questions:
            level = question.get('bloom_level')
            if not level:
                # Classify if not already classified
                level = self.classify_question(question.get('question', ''))
                question['bloom_level'] = level
            
            distribution[level] = distribution.get(level, 0) + 1
        
        return distribution
    
    def validate_distribution(
        self,
        questions: List[Dict[str, Any]],
        target_distribution: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Validate if question distribution matches target distribution
        
        Args:
            questions: List of questions
            target_distribution: Target distribution percentages
        
        Returns:
            Validation result with gaps and recommendations
        """
        current_dist = self.get_level_distribution(questions)
        total_questions = len(questions)
        
        gaps = {}
        for level, target_pct in target_distribution.items():
            current_count = current_dist.get(level.lower(), 0)
            current_pct = (current_count / total_questions * 100) if total_questions > 0 else 0
            gap = target_pct - current_pct
            
            gaps[level] = {
                'target_percentage': target_pct,
                'current_percentage': current_pct,
                'gap': gap,
                'needs_more': gap > 5,  # More than 5% gap
                'needs_less': gap < -5
            }
        
        return {
            'current_distribution': current_dist,
            'gaps': gaps,
            'is_balanced': all(abs(g['gap']) <= 5 for g in gaps.values())
        }
    
    @staticmethod
    def get_level_description(level: str) -> str:
        """Get description of a Bloom's taxonomy level"""
        descriptions = {
            'remember': 'Recall facts and basic concepts. Students retrieve relevant knowledge from long-term memory.',
            'understand': 'Explain ideas or concepts. Students construct meaning from instructional messages.',
            'apply': 'Use information in new situations. Students carry out or use a procedure in a given situation.',
            'analyze': 'Draw connections among ideas. Students break material into parts and determine relationships.',
            'evaluate': 'Justify a stand or decision. Students make judgments based on criteria and standards.',
            'create': 'Produce new or original work. Students put elements together to form a coherent whole.'
        }
        return descriptions.get(level.lower(), 'Unknown level')
    
    @staticmethod
    def get_cognitive_process_verbs(level: str) -> List[str]:
        """Get typical verbs for a Bloom's level"""
        return BloomClassifier.BLOOM_KEYWORDS.get(level.lower(), [])
