"""
Priority Calculator Module
Calculates priority scores for topics based on syllabus weight and historical frequency
"""

import re
from typing import List, Dict, Any
import numpy as np


class PriorityCalculator:
    """
    Calculate priority scores for syllabus topics
    Formula: P = w1 * S + w2 * F
    Where: S = Syllabus weight, F = Historical frequency
    """
    
    def __init__(self, syllabus_weight: float = 0.6, frequency_weight: float = 0.4):
        """
        Initialize Priority Calculator
        
        Args:
            syllabus_weight: Weight for syllabus coverage (default: 0.6)
            frequency_weight: Weight for historical frequency (default: 0.4)
        """
        self.w1 = syllabus_weight
        self.w2 = frequency_weight
        
        # Default frequency data (can be updated from past papers)
        self.historical_frequencies = {}
    
    def calculate_priorities(self, syllabus_text: str) -> List[Dict[str, Any]]:
        """
        Calculate priority scores for all topics in syllabus
        
        Args:
            syllabus_text: Preprocessed syllabus text
        
        Returns:
            List of topics with priority scores
        """
        # Extract topics from syllabus
        topics = self._extract_topics(syllabus_text)
        
        # Calculate syllabus weights
        topics_with_weights = self._calculate_syllabus_weights(topics, syllabus_text)
        
        # Add historical frequencies
        topics_with_freq = self._add_historical_frequencies(topics_with_weights)
        
        # Calculate final priority scores
        topics_with_priority = self._calculate_final_priorities(topics_with_freq)
        
        # Sort by priority (descending)
        topics_with_priority.sort(key=lambda x: x['priority'], reverse=True)
        
        return topics_with_priority
    
    def _extract_topics(self, syllabus_text: str) -> List[str]:
        """Extract topic names from syllabus"""
        topics = []
        
        # Pattern 1: Numbered topics (1. Topic, 2. Topic)
        numbered_pattern = r'^\s*\d+[\.\)]\s*([A-Z][^\n]+?)(?:\s*[-–—]\s*|\s*\n)'
        numbered_topics = re.findall(numbered_pattern, syllabus_text, re.MULTILINE)
        topics.extend(numbered_topics)
        
        # Pattern 2: Bulleted topics
        bullet_pattern = r'^\s*[•●■▪]\s*([A-Z][^\n]+?)(?:\s*\n)'
        bullet_topics = re.findall(bullet_pattern, syllabus_text, re.MULTILINE)
        topics.extend(bullet_topics)
        
        # Pattern 3: Unit/Module headers
        unit_pattern = r'(?:Unit|Module|Chapter|Section)\s+\d+[:\s]+([A-Z][^\n]+)'
        unit_topics = re.findall(unit_pattern, syllabus_text, re.IGNORECASE)
        topics.extend(unit_topics)
        
        # Clean and deduplicate
        topics = [self._clean_topic_name(t) for t in topics]
        topics = list(dict.fromkeys(topics))  # Remove duplicates while preserving order
        
        # If no topics found, split by major sections
        if len(topics) == 0:
            topics = self._extract_topics_by_sections(syllabus_text)
        
        return topics
    
    def _clean_topic_name(self, topic: str) -> str:
        """Clean topic name"""
        # Remove extra whitespace
        topic = ' '.join(topic.split())
        
        # Remove trailing punctuation
        topic = topic.rstrip('.:;,')
        
        # Capitalize properly
        topic = topic.title()
        
        return topic
    
    def _extract_topics_by_sections(self, text: str) -> List[str]:
        """Extract topics by identifying major sections"""
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Get first sentence of each paragraph as potential topic
        topics = []
        for para in paragraphs[:10]:  # Limit to first 10 paragraphs
            # Get first sentence
            sentences = re.split(r'[.!?]', para)
            if sentences and len(sentences[0]) > 10:
                topic = sentences[0].strip()
                if len(topic) < 100:  # Reasonable topic length
                    topics.append(topic)
        
        return topics
    
    def _calculate_syllabus_weights(
        self, topics: List[str], syllabus_text: str
    ) -> List[Dict[str, Any]]:
        """Calculate syllabus weight for each topic"""
        topics_data = []
        
        for topic in topics:
            # Count occurrences of topic in syllabus
            occurrences = self._count_topic_occurrences(topic, syllabus_text)
            
            # Calculate weight based on occurrences and context
            weight = self._calculate_weight(topic, syllabus_text, occurrences)
            
            topics_data.append({
                'name': topic,
                'topic': topic,  # Alias for compatibility
                'syllabus_weight': weight,
                'occurrences': occurrences
            })
        
        # Normalize weights to sum to 1.0
        total_weight = sum(t['syllabus_weight'] for t in topics_data)
        if total_weight > 0:
            for topic in topics_data:
                topic['syllabus_weight'] = topic['syllabus_weight'] / total_weight
        
        return topics_data
    
    def _count_topic_occurrences(self, topic: str, text: str) -> int:
        """Count how many times a topic appears in text"""
        # Create regex pattern (case-insensitive, word boundaries)
        pattern = r'\b' + re.escape(topic) + r'\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(matches)
    
    def _calculate_weight(self, topic: str, syllabus_text: str, occurrences: int) -> float:
        """Calculate weight for a topic based on various factors"""
        weight = 0.0
        
        # Factor 1: Number of occurrences
        weight += occurrences * 0.3
        
        # Factor 2: Position in syllabus (earlier = more important)
        position = syllabus_text.lower().find(topic.lower())
        if position != -1:
            position_score = 1.0 - (position / len(syllabus_text))
            weight += position_score * 0.2
        
        # Factor 3: Context indicators (words like "important", "key", etc.)
        context_keywords = [
            'important', 'key', 'essential', 'fundamental', 'core', 
            'critical', 'major', 'primary', 'main'
        ]
        
        # Get context around topic
        topic_index = syllabus_text.lower().find(topic.lower())
        if topic_index != -1:
            start = max(0, topic_index - 100)
            end = min(len(syllabus_text), topic_index + len(topic) + 100)
            context = syllabus_text[start:end].lower()
            
            for keyword in context_keywords:
                if keyword in context:
                    weight += 0.5
        
        # Factor 4: Length of topic (longer topics might be more complex)
        word_count = len(topic.split())
        if word_count > 3:
            weight += 0.3
        
        return max(0.1, weight)  # Minimum weight of 0.1
    
    def _add_historical_frequencies(
        self, topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add historical frequency data to topics"""
        for topic in topics:
            topic_name = topic['name']
            
            # Get historical frequency (default to 0.5 if not found)
            frequency = self.historical_frequencies.get(topic_name, 0.5)
            topic['historical_frequency'] = frequency
        
        return topics
    
    def _calculate_final_priorities(
        self, topics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate final priority scores"""
        for topic in topics:
            S = topic['syllabus_weight']
            F = topic['historical_frequency']
            
            # Priority formula: P = w1 * S + w2 * F
            priority = self.w1 * S + self.w2 * F
            
            topic['priority'] = priority
        
        # Normalize priorities to 0-1 range
        priorities = [t['priority'] for t in topics]
        if priorities:
            min_p = min(priorities)
            max_p = max(priorities)
            
            if max_p > min_p:
                for topic in topics:
                    topic['priority'] = (topic['priority'] - min_p) / (max_p - min_p)
        
        return topics
    
    def update_historical_frequencies(self, past_papers_data: List[Dict[str, Any]]):
        """
        Update historical frequency data from past papers
        
        Args:
            past_papers_data: List of past papers with topic occurrences
        """
        topic_counts = {}
        total_papers = len(past_papers_data)
        
        if total_papers == 0:
            return
        
        # Count topics across all past papers
        for paper in past_papers_data:
            topics = paper.get('topics', [])
            for topic in topics:
                topic_name = topic if isinstance(topic, str) else topic.get('name', '')
                topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
        
        # Calculate frequencies
        for topic, count in topic_counts.items():
            self.historical_frequencies[topic] = count / total_papers
    
    def get_top_priority_topics(self, topics: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N priority topics
        
        Args:
            topics: List of topics with priorities
            n: Number of top topics to return
        
        Returns:
            Top N topics
        """
        sorted_topics = sorted(topics, key=lambda x: x['priority'], reverse=True)
        return sorted_topics[:n]
