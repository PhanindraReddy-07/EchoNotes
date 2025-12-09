"""
Text Preprocessor Module
Uses NLTK for text cleaning and tokenization.
"""

import re
from typing import List

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class Preprocessor:
    """Text preprocessing for speech transcripts."""
    
    # Common filler words in speech
    FILLER_WORDS = {
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean',
        'basically', 'actually', 'literally', 'so', 'well',
        'kind of', 'sort of', 'right', 'okay', 'ok'
    }
    
    # Multi-word fillers (processed separately)
    MULTI_WORD_FILLERS = [
        'you know', 'i mean', 'kind of', 'sort of'
    ]
    
    def __init__(self, settings):
        self.settings = settings
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        if not NLTK_AVAILABLE:
            return
        
        required = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
        for package in required:
            try:
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' 
                              else f'corpora/{package}' if package == 'stopwords'
                              else f'taggers/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except:
                    pass
    
    def clean_text(self, text: str) -> str:
        """
        Clean transcript text.
        
        - Remove filler words
        - Fix spacing and punctuation
        - Capitalize sentences
        """
        if not text:
            return ""
        
        # Lowercase for processing
        text = text.lower()
        
        # Remove multi-word fillers first
        for filler in self.MULTI_WORD_FILLERS:
            text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
        
        # Remove single-word fillers
        single_fillers = self.FILLER_WORDS - set(self.MULTI_WORD_FILLERS)
        for filler in single_fillers:
            # Only remove if standalone (not part of larger word)
            text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
        
        # Capitalize sentences
        text = self._capitalize_sentences(text)
        
        return text
    
    def _capitalize_sentences(self, text: str) -> str:
        """Capitalize first letter of each sentence."""
        if not text:
            return ""
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'([.!?]+\s*)', text)
        
        result = []
        for i, part in enumerate(sentences):
            if part and not re.match(r'^[.!?\s]+$', part):
                # Capitalize first letter
                part = part[0].upper() + part[1:] if part else part
            result.append(part)
        
        return ''.join(result)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback: simple regex split
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """Split text into words."""
        if not text:
            return []
        
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except:
                pass
        
        # Fallback: simple split
        return text.split()
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove common stopwords."""
        if NLTK_AVAILABLE:
            try:
                stop_words = set(stopwords.words('english'))
                return [w for w in words if w.lower() not in stop_words]
            except:
                pass
        
        # Minimal fallback stopwords
        basic_stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                       'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                       'as', 'into', 'through', 'during', 'before', 'after'}
        return [w for w in words if w.lower() not in basic_stops]
    
    def get_word_frequency(self, text: str) -> dict:
        """Get word frequency distribution."""
        words = self.tokenize_words(text.lower())
        words = self.remove_stopwords(words)
        
        # Remove punctuation-only tokens
        words = [w for w in words if re.match(r'\w+', w)]
        
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
