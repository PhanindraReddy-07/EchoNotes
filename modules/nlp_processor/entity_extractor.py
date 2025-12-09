"""
Entity Extractor Module
Named Entity Recognition using NLTK.
"""

import re
from typing import Dict, List
from datetime import datetime

try:
    import nltk
    from nltk import pos_tag, word_tokenize, ne_chunk
    from nltk.tree import Tree
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class EntityExtractor:
    """Extract named entities from text using NLTK."""
    
    # Date patterns for regex extraction
    DATE_PATTERNS = [
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # MM/DD/YYYY
        r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',    # YYYY-MM-DD
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b',
        r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b(today|tomorrow|yesterday)\b',
        r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'\b(next|last)\s+(week|month|year)\b',
    ]
    
    # Time patterns
    TIME_PATTERNS = [
        r'\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b',
        r'\b(\d{1,2}\s*(?:AM|PM|am|pm))\b',
    ]
    
    # Money patterns
    MONEY_PATTERNS = [
        r'\$[\d,]+(?:\.\d{2})?',
        r'[\d,]+\s*(?:dollars|USD|euros|EUR|pounds|GBP)',
    ]
    
    def __init__(self, settings):
        self.settings = settings
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data."""
        if not NLTK_AVAILABLE:
            return
        
        required = [
            'punkt',
            'averaged_perceptron_tagger', 
            'maxent_ne_chunker',
            'words'
        ]
        
        for package in required:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt'
                              else f'taggers/{package}' if 'tagger' in package
                              else f'chunkers/{package}' if 'chunker' in package
                              else f'corpora/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except:
                    pass
    
    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all named entities from text.
        
        Returns dict with keys: persons, organizations, locations, dates, times, money
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'times': [],
            'money': [],
        }
        
        if not text:
            return entities
        
        # NLTK-based extraction
        if NLTK_AVAILABLE:
            nltk_entities = self._nltk_extract(text)
            for key in ['persons', 'organizations', 'locations']:
                entities[key].extend(nltk_entities.get(key, []))
        
        # Regex-based extraction for dates, times, money
        entities['dates'].extend(self._extract_dates(text))
        entities['times'].extend(self._extract_times(text))
        entities['money'].extend(self._extract_money(text))
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _nltk_extract(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using NLTK NER."""
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
        }
        
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity chunking
            tree = ne_chunk(pos_tags)
            
            # Extract entities from tree
            for subtree in tree:
                if isinstance(subtree, Tree):
                    entity_text = ' '.join(word for word, tag in subtree.leaves())
                    entity_type = subtree.label()
                    
                    if entity_type == 'PERSON':
                        entities['persons'].append(entity_text)
                    elif entity_type == 'ORGANIZATION':
                        entities['organizations'].append(entity_text)
                    elif entity_type in ('GPE', 'LOCATION'):
                        entities['locations'].append(entity_text)
        except Exception as e:
            print(f"NLTK NER error: {e}")
        
        return entities
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract date mentions using regex."""
        dates = []
        for pattern in self.DATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    dates.append(' '.join(m for m in match if m))
                else:
                    dates.append(match)
        return dates
    
    def _extract_times(self, text: str) -> List[str]:
        """Extract time mentions using regex."""
        times = []
        for pattern in self.TIME_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)
        return times
    
    def _extract_money(self, text: str) -> List[str]:
        """Extract monetary values using regex."""
        money = []
        for pattern in self.MONEY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            money.extend(matches)
        return money
    
    def extract_action_items(self, text: str) -> List[str]:
        """
        Extract potential action items from text.
        Looks for imperative sentences and task-related phrases.
        """
        action_patterns = [
            r'(?:need to|must|should|have to|going to|will)\s+([^.!?]+)',
            r'(?:action item|todo|task|reminder):\s*([^.!?]+)',
            r'(?:please|kindly)\s+([^.!?]+)',
            r'(?:make sure|ensure|remember to)\s+([^.!?]+)',
            r'(?:follow up|schedule|send|create|prepare|review)\s+([^.!?]+)',
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                action = match.strip()
                if len(action) > 10 and len(action) < 200:
                    actions.append(action.capitalize())
        
        return list(set(actions))
