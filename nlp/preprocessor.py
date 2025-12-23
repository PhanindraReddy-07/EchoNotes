"""
Text Preprocessor - Clean and prepare transcripts for NLP
==========================================================
Handles filler removal, sentence segmentation, and text normalization
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """Container for preprocessed text"""
    original: str
    cleaned: str
    sentences: List[str]
    word_count: int
    filler_count: int
    fillers_removed: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'original_length': len(self.original),
            'cleaned_length': len(self.cleaned),
            'sentence_count': len(self.sentences),
            'word_count': self.word_count,
            'fillers_removed': self.filler_count
        }


class TextPreprocessor:
    """
    Preprocessor for cleaning ASR transcripts
    
    Features:
    - Filler word removal (um, uh, like, you know, etc.)
    - False start removal
    - Sentence boundary detection
    - Text normalization
    - Repeated word handling
    
    Usage:
        preprocessor = TextPreprocessor()
        result = preprocessor.process(transcript_text)
        print(result.cleaned)
    """
    
    # Common filler words and phrases
    DEFAULT_FILLERS = [
        # Single word fillers
        'um', 'uh', 'er', 'ah', 'eh', 'hmm', 'hm', 'mm',
        'like', 'basically', 'actually', 'literally', 'honestly',
        'obviously', 'clearly', 'really', 'totally', 'definitely',
        'anyway', 'anyways', 'whatever', 'well',
        
        # Multi-word fillers
        'you know', 'i mean', 'you see', 'kind of', 'sort of',
        'i guess', 'i think', 'i believe', 'in a sense',
        'at the end of the day', 'to be honest', 'to be fair',
        'as i said', 'as i mentioned', 'if you will',
        'so to speak', 'more or less', 'or something',
        'and stuff', 'and things', 'and all that',
    ]
    
    # Indian English fillers
    INDIAN_FILLERS = [
        'actually', 'basically', 'obviously', 'simply',
        'na', 'no', 'ya', 'haan', 'ok so', 'so basically',
        'what happened is', 'the thing is', 'see the thing is',
        'i will tell you', 'let me tell you',
    ]
    
    def __init__(
        self,
        remove_fillers: bool = True,
        custom_fillers: Optional[List[str]] = None,
        include_indian_fillers: bool = True,
        min_sentence_words: int = 3
    ):
        """
        Initialize preprocessor
        
        Args:
            remove_fillers: Whether to remove filler words
            custom_fillers: Additional filler words/phrases
            include_indian_fillers: Include Indian English fillers
            min_sentence_words: Minimum words for a valid sentence
        """
        self.remove_fillers = remove_fillers
        self.min_sentence_words = min_sentence_words
        
        # Build filler list
        self.fillers = set(self.DEFAULT_FILLERS)
        if include_indian_fillers:
            self.fillers.update(self.INDIAN_FILLERS)
        if custom_fillers:
            self.fillers.update(f.lower() for f in custom_fillers)
        
        # Sort by length (longest first) for proper replacement
        self.fillers_sorted = sorted(self.fillers, key=len, reverse=True)
        
        # Compile regex patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        # Pattern for repeated words (e.g., "the the", "I I I")
        self.repeated_word_pattern = re.compile(r'\b(\w+)(\s+\1)+\b', re.IGNORECASE)
        
        # Pattern for multiple spaces
        self.multi_space_pattern = re.compile(r'\s+')
        
        # Pattern for sentence boundaries
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        # Pattern for false starts (e.g., "I wa- I was going")
        self.false_start_pattern = re.compile(r'\b\w+\-\s*')
    
    def process(self, text: str) -> ProcessedText:
        """
        Process and clean transcript text
        
        Args:
            text: Raw transcript text
            
        Returns:
            ProcessedText with cleaned text and metadata
        """
        original = text
        cleaned = text
        fillers_found = []
        
        # Step 1: Normalize whitespace
        cleaned = self.multi_space_pattern.sub(' ', cleaned).strip()
        
        # Step 2: Remove false starts
        cleaned = self.false_start_pattern.sub('', cleaned)
        
        # Step 3: Remove filler words/phrases
        if self.remove_fillers:
            cleaned, fillers_found = self._remove_fillers(cleaned)
        
        # Step 4: Handle repeated words
        cleaned = self._remove_repeated_words(cleaned)
        
        # Step 5: Clean up punctuation
        cleaned = self._clean_punctuation(cleaned)
        
        # Step 6: Normalize whitespace again
        cleaned = self.multi_space_pattern.sub(' ', cleaned).strip()
        
        # Step 7: Segment into sentences
        sentences = self._segment_sentences(cleaned)
        
        # Count words
        word_count = len(cleaned.split())
        
        return ProcessedText(
            original=original,
            cleaned=cleaned,
            sentences=sentences,
            word_count=word_count,
            filler_count=len(fillers_found),
            fillers_removed=fillers_found
        )
    
    def _remove_fillers(self, text: str) -> Tuple[str, List[str]]:
        """Remove filler words and phrases"""
        fillers_found = []
        result = text.lower()
        
        for filler in self.fillers_sorted:
            # Create pattern that matches whole words/phrases
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, result, re.IGNORECASE)
            if matches:
                fillers_found.extend(matches)
                result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        # Restore original case for remaining words
        # (simplified: just capitalize first letter of sentences)
        result = self._restore_case(result)
        
        return result, fillers_found
    
    def _restore_case(self, text: str) -> str:
        """Restore proper capitalization"""
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after sentence endings
        def capitalize_match(match):
            return match.group(0).upper()
        
        text = re.sub(r'(?<=[.!?]\s)([a-z])', capitalize_match, text)
        
        # Capitalize 'I'
        text = re.sub(r'\bi\b', 'I', text)
        
        return text
    
    def _remove_repeated_words(self, text: str) -> str:
        """Remove immediately repeated words"""
        return self.repeated_word_pattern.sub(r'\1', text)
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean up punctuation issues"""
        # Remove multiple punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        
        # Remove orphan punctuation
        text = re.sub(r'^\s*[.!?,;:]\s*', '', text)
        
        return text
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        # Split on sentence boundaries
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = []
        for sent in raw_sentences:
            sent = sent.strip()
            # Filter out very short sentences
            if len(sent.split()) >= self.min_sentence_words:
                # Ensure sentence ends with punctuation
                if sent and sent[-1] not in '.!?':
                    sent += '.'
                sentences.append(sent)
        
        return sentences
    
    def clean_for_nlp(self, text: str) -> str:
        """
        Quick clean for NLP processing (minimal processing)
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Just normalize whitespace and fix basic issues
        text = self.multi_space_pattern.sub(' ', text).strip()
        text = self._remove_repeated_words(text)
        return text
    
    def get_statistics(self, text: str) -> Dict:
        """
        Get text statistics without full processing
        
        Args:
            text: Input text
            
        Returns:
            Statistics dictionary
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Count fillers
        filler_count = 0
        text_lower = text.lower()
        for filler in self.fillers:
            filler_count += len(re.findall(r'\b' + re.escape(filler) + r'\b', text_lower))
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(1, len(sentences)),
            'filler_count': filler_count,
            'filler_ratio': filler_count / max(1, len(words))
        }
