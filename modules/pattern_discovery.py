"""
Pattern Discovery Module
Discovers patterns and trends from past examination papers
"""

from typing import List, Dict, Any, Optional
import re
from collections import Counter
import numpy as np


class PatternDiscovery:
    """
    Analyze past papers to discover patterns and trends
    """
    
    def __init__(self):
        """Initialize Pattern Discovery"""
        self.patterns = {}
        self.topic_frequencies = {}
        self.question_types = {}
    
    def analyze(self, syllabus_text: str, past_papers: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze syllabus and past papers to discover patterns
        
        Args:
            syllabus_text: Syllabus text
            past_papers: List of past paper data
        
        Returns:
            Discovered patterns
        """
        patterns = {
            'topic_analysis': self._analyze_topics(syllabus_text),
            'structure_analysis': self._analyze_structure(syllabus_text),
            'difficulty_distribution': {}
        }
        
        if past_papers:
            patterns['historical_patterns'] = self._analyze_past_papers(past_papers)
            patterns['topic_frequencies'] = self._calculate_topic_frequencies(past_papers)
            patterns['question_type_distribution'] = self._analyze_question_types(past_papers)
        
        self.patterns = patterns
        return patterns
    
    def _analyze_topics(self, syllabus_text: str) -> Dict[str, Any]:
        """Analyze topics in syllabus"""
        # Extract potential topics
        lines = [line.strip() for line in syllabus_text.split('\n') if line.strip()]
        
        # Identify topic indicators
        topic_patterns = [
            r'^\d+[\.\)]\s+(.+)$',  # Numbered topics
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:',  # Capitalized topics with colon
            r'(?i)^(?:unit|module|chapter|section)\s+\d+[:\s]+(.+)$'
        ]
        
        topics = []
        for line in lines:
            for pattern in topic_patterns:
                match = re.match(pattern, line)
                if match:
                    topic = match.group(1) if match.lastindex else line
                    topics.append(topic.strip())
                    break
        
        # Calculate topic statistics
        return {
            'count': len(topics),
            'topics': topics[:20],  # Top 20 topics
            'avg_words_per_topic': np.mean([len(t.split()) for t in topics]) if topics else 0
        }
    
    def _analyze_structure(self, syllabus_text: str) -> Dict[str, Any]:
        """Analyze structural patterns in syllabus"""
        lines = syllabus_text.split('\n')
        
        # Count different structural elements
        numbered_items = len(re.findall(r'^\s*\d+[\.\)]', syllabus_text, re.MULTILINE))
        bulleted_items = len(re.findall(r'^\s*[•●■▪-]', syllabus_text, re.MULTILINE))
        units = len(re.findall(r'(?i)\b(?:unit|module|chapter)\s+\d+', syllabus_text))
        
        return {
            'numbered_items': numbered_items,
            'bulleted_items': bulleted_items,
            'units_or_modules': units,
            'total_lines': len(lines),
            'non_empty_lines': len([l for l in lines if l.strip()])
        }
    
    def _analyze_past_papers(self, past_papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in past papers"""
        if not past_papers:
            return {}
        
        total_papers = len(past_papers)
        
        # Collect statistics
        total_questions = []
        marks_distributions = []
        bloom_distributions = []
        
        for paper in past_papers:
            questions = paper.get('questions', [])
            total_questions.append(len(questions))
            
            # Marks distribution
            marks = [q.get('marks', 0) for q in questions if isinstance(q, dict)]
            if marks:
                marks_distributions.append({
                    'min': min(marks),
                    'max': max(marks),
                    'avg': np.mean(marks),
                    'total': sum(marks)
                })
            
            # Bloom's level distribution
            bloom_levels = [q.get('bloom_level', '') for q in questions if isinstance(q, dict)]
            if bloom_levels:
                bloom_counter = Counter(bloom_levels)
                bloom_distributions.append(dict(bloom_counter))
        
        # Aggregate patterns
        patterns = {
            'paper_count': total_papers,
            'avg_questions_per_paper': np.mean(total_questions) if total_questions else 0,
            'question_count_range': {
                'min': min(total_questions) if total_questions else 0,
                'max': max(total_questions) if total_questions else 0
            }
        }
        
        if marks_distributions:
            patterns['marks_pattern'] = {
                'avg_total_marks': np.mean([m['total'] for m in marks_distributions]),
                'common_question_marks': self._find_common_marks(marks_distributions)
            }
        
        if bloom_distributions:
            patterns['bloom_pattern'] = self._aggregate_bloom_distribution(bloom_distributions)
        
        return patterns
    
    def _calculate_topic_frequencies(self, past_papers: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate how frequently topics appear in past papers"""
        topic_counter = Counter()
        total_papers = len(past_papers)
        
        for paper in past_papers:
            questions = paper.get('questions', [])
            
            for q in questions:
                if not isinstance(q, dict):
                    continue
                
                topic = q.get('topic', '')
                if topic:
                    topic_counter[topic] += 1
        
        # Convert to frequencies (0-1)
        frequencies = {}
        for topic, count in topic_counter.items():
            frequencies[topic] = count / total_papers if total_papers > 0 else 0
        
        # Sort by frequency
        sorted_frequencies = dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_frequencies
    
    def _analyze_question_types(self, past_papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution of question types"""
        type_counter = Counter()
        
        for paper in past_papers:
            questions = paper.get('questions', [])
            
            for q in questions:
                if not isinstance(q, dict):
                    continue
                
                # Classify question type based on marks or keywords
                marks = q.get('marks', 0)
                question_text = q.get('question', '').lower()
                
                if marks <= 2:
                    q_type = 'Very Short Answer'
                elif marks <= 5:
                    q_type = 'Short Answer'
                elif marks <= 10:
                    q_type = 'Medium Answer'
                else:
                    q_type = 'Long Answer'
                
                # Further classify by keywords
                if 'define' in question_text or 'what is' in question_text:
                    q_type += ' (Definition)'
                elif 'explain' in question_text:
                    q_type += ' (Explanation)'
                elif 'compare' in question_text or 'difference' in question_text:
                    q_type += ' (Comparison)'
                elif 'derive' in question_text or 'prove' in question_text:
                    q_type += ' (Derivation/Proof)'
                
                type_counter[q_type] += 1
        
        return dict(type_counter)
    
    def _find_common_marks(self, marks_distributions: List[Dict[str, Any]]) -> List[int]:
        """Find most commonly used marks for questions"""
        all_marks = []
        
        for dist in marks_distributions:
            # This is simplified; in real implementation, 
            # you'd track individual question marks
            pass
        
        # Common exam marks patterns
        return [2, 3, 5, 8, 10, 15]
    
    def _aggregate_bloom_distribution(self, bloom_distributions: List[Dict[str, int]]) -> Dict[str, float]:
        """Aggregate Bloom's taxonomy distribution across papers"""
        total_counter = Counter()
        
        for dist in bloom_distributions:
            total_counter.update(dist)
        
        # Convert to percentages
        total = sum(total_counter.values())
        percentages = {}
        
        for level, count in total_counter.items():
            if total > 0:
                percentages[level] = (count / total) * 100
        
        return percentages
    
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations based on discovered patterns
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not self.patterns:
            return ["No patterns analyzed yet. Please run analyze() first."]
        
        # Check topic coverage
        topic_analysis = self.patterns.get('topic_analysis', {})
        topic_count = topic_analysis.get('count', 0)
        
        if topic_count < 5:
            recommendations.append(
                "Low number of topics detected. Consider adding more specific topics to syllabus."
            )
        elif topic_count > 30:
            recommendations.append(
                "High number of topics. Consider grouping related topics for better organization."
            )
        
        # Check historical patterns
        historical = self.patterns.get('historical_patterns', {})
        if historical:
            avg_questions = historical.get('avg_questions_per_paper', 0)
            
            if avg_questions > 0:
                recommendations.append(
                    f"Historical average: {avg_questions:.0f} questions per paper. "
                    f"Aim for similar count for consistency."
                )
            
            # Bloom's distribution recommendations
            bloom_pattern = historical.get('bloom_pattern', {})
            if bloom_pattern:
                # Check if higher-order thinking questions are present
                higher_order = (
                    bloom_pattern.get('analyze', 0) +
                    bloom_pattern.get('evaluate', 0) +
                    bloom_pattern.get('create', 0)
                )
                
                if higher_order < 20:  # Less than 20%
                    recommendations.append(
                        "Consider adding more higher-order thinking questions "
                        "(Analyze, Evaluate, Create levels) for better cognitive assessment."
                    )
        
        # Topic frequency recommendations
        topic_freq = self.patterns.get('topic_frequencies', {})
        if topic_freq:
            # Identify hot topics (frequently appeared)
            hot_topics = [topic for topic, freq in topic_freq.items() if freq > 0.7]
            
            if hot_topics:
                recommendations.append(
                    f"High-frequency topics detected: {', '.join(hot_topics[:3])}. "
                    f"Ensure adequate coverage of these topics."
                )
        
        if not recommendations:
            recommendations.append("Patterns look balanced. Proceed with generation.")
        
        return recommendations
    
    def export_patterns(self) -> Dict[str, Any]:
        """Export discovered patterns for storage"""
        return {
            'patterns': self.patterns,
            'topic_frequencies': self.topic_frequencies,
            'question_types': self.question_types,
            'timestamp': str(np.datetime64('now'))
        }
    
    def import_patterns(self, patterns_data: Dict[str, Any]):
        """Import previously discovered patterns"""
        self.patterns = patterns_data.get('patterns', {})
        self.topic_frequencies = patterns_data.get('topic_frequencies', {})
        self.question_types = patterns_data.get('question_types', {})
