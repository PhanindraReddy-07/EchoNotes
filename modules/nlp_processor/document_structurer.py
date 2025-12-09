"""
Document Structurer Module
Organizes processed content into structured document format.
"""

import re
from typing import Dict, List, Any
from datetime import datetime


class DocumentStructurer:
    """Structures processed content into document format."""
    
    # Keywords for document type detection
    MEETING_KEYWORDS = [
        'meeting', 'agenda', 'minutes', 'attendees', 'action items',
        'discussed', 'agreed', 'decision', 'next steps', 'follow up'
    ]
    
    LECTURE_KEYWORDS = [
        'lecture', 'class', 'lesson', 'chapter', 'topic', 'today we',
        'learn', 'understand', 'concept', 'example', 'important'
    ]
    
    INTERVIEW_KEYWORDS = [
        'interview', 'candidate', 'position', 'experience', 'skills',
        'tell me about', 'why do you', 'strengths', 'weaknesses'
    ]
    
    def __init__(self, settings):
        self.settings = settings
    
    def structure(self, text: str, sentences: List[str], 
                  entities: Dict[str, List[str]], summary: str,
                  metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Structure all processed content into document format.
        
        Returns:
            Dict with structured document data
        """
        doc_type = self._detect_document_type(text)
        
        document = {
            'metadata': {
                'type': doc_type,
                'generated_at': datetime.now().isoformat(),
                'word_count': len(text.split()),
                'sentence_count': len(sentences),
                **(metadata or {})
            },
            'title': self._generate_title(text, doc_type),
            'summary': summary,
            'content': {
                'full_text': text,
                'sections': self._create_sections(sentences, doc_type),
            },
            'entities': entities,
            'action_items': self._extract_action_items(text),
            'key_points': self._extract_key_points(sentences),
        }
        
        # Add type-specific fields
        if doc_type == 'meeting':
            document['attendees'] = entities.get('persons', [])
            document['dates_mentioned'] = entities.get('dates', [])
        elif doc_type == 'lecture':
            document['topics'] = self._extract_topics(text)
        
        return document
    
    def _detect_document_type(self, text: str) -> str:
        """Detect document type based on content."""
        text_lower = text.lower()
        
        # Count keyword matches
        scores = {
            'meeting': sum(1 for kw in self.MEETING_KEYWORDS if kw in text_lower),
            'lecture': sum(1 for kw in self.LECTURE_KEYWORDS if kw in text_lower),
            'interview': sum(1 for kw in self.INTERVIEW_KEYWORDS if kw in text_lower),
        }
        
        # Return type with highest score, or 'general' if no clear match
        max_type = max(scores, key=scores.get)
        if scores[max_type] >= 2:
            return max_type
        return 'general'
    
    def _generate_title(self, text: str, doc_type: str) -> str:
        """Generate a title for the document."""
        # Get first meaningful sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if sentences:
            first = sentences[0][:100]
            # Clean and truncate
            first = re.sub(r'\s+', ' ', first).strip()
            if len(first) > 60:
                first = first[:57] + '...'
            return first
        
        # Fallback titles
        titles = {
            'meeting': 'Meeting Notes',
            'lecture': 'Lecture Notes',
            'interview': 'Interview Notes',
            'general': 'Notes'
        }
        return titles.get(doc_type, 'Notes')
    
    def _create_sections(self, sentences: List[str], doc_type: str) -> List[Dict]:
        """Organize sentences into logical sections."""
        if not sentences:
            return []
        
        # Simple sectioning: group by paragraph breaks or every N sentences
        sections = []
        current_section = {
            'title': 'Main Content',
            'content': []
        }
        
        for i, sentence in enumerate(sentences):
            current_section['content'].append(sentence)
            
            # Create new section every 5-7 sentences or at topic shifts
            if len(current_section['content']) >= 6:
                sections.append(current_section)
                section_num = len(sections) + 1
                current_section = {
                    'title': f'Section {section_num}',
                    'content': []
                }
        
        # Add remaining content
        if current_section['content']:
            sections.append(current_section)
        
        # If only one section, just call it main content
        if len(sections) == 1:
            sections[0]['title'] = 'Content'
        
        return sections
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text."""
        patterns = [
            r'(?:need to|must|should|have to)\s+([^.!?]+)',
            r'(?:action item|todo|task):\s*([^.!?]+)',
            r'(?:follow up|schedule|send|create|review)\s+([^.!?]+)',
            r'(?:please|kindly)\s+([^.!?]+)',
        ]
        
        actions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                action = match.strip().capitalize()
                if 10 < len(action) < 150:
                    actions.append(action)
        
        return list(set(actions))[:10]  # Limit to 10 items
    
    def _extract_key_points(self, sentences: List[str], max_points: int = 5) -> List[str]:
        """Extract key points from sentences."""
        if not sentences:
            return []
        
        # Look for sentences with key indicators
        key_indicators = [
            'important', 'key', 'main', 'critical', 'essential',
            'remember', 'note that', 'in summary', 'to conclude'
        ]
        
        key_sentences = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(ind in lower for ind in key_indicators):
                key_sentences.append(sentence)
        
        # If no explicit key points, take first sentence from each section
        if not key_sentences and len(sentences) > 3:
            step = max(1, len(sentences) // max_points)
            key_sentences = [sentences[i] for i in range(0, len(sentences), step)]
        
        return key_sentences[:max_points]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics (for lectures)."""
        # Look for topic indicators
        patterns = [
            r'(?:topic|subject|about|discussing|covering):\s*([^.!?]+)',
            r'(?:first|second|third|next|finally),?\s*([^.!?]+)',
        ]
        
        topics = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            topics.extend(matches)
        
        return [t.strip().capitalize() for t in topics[:5]]
