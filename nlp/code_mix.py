"""
Code-Mixed Language Handler
============================
Detects and processes Telugu-English and Hindi-English code-mixed speech

This is a KEY NOVEL COMPONENT for Indian language support.
Real Indian conversations often mix languages mid-sentence:
"Meeting tomorrow ko postpone kar diya" (Hindi-English)
"Nenu report prepare chesthanu by Friday" (Telugu-English)
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    HINDI = "hi"
    TELUGU = "te"
    MIXED_HI_EN = "hi-en"
    MIXED_TE_EN = "te-en"
    UNKNOWN = "unknown"


@dataclass
class LanguageSegment:
    """A segment of text in a specific language"""
    text: str
    language: Language
    start_pos: int
    end_pos: int
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'language': self.language.value,
            'confidence': round(self.confidence, 3)
        }


@dataclass
class CodeMixResult:
    """Result of code-mix analysis"""
    original_text: str
    primary_language: Language
    segments: List[LanguageSegment]
    is_code_mixed: bool
    language_distribution: Dict[str, float]  # Percentage per language
    transliterated_text: Optional[str] = None  # Roman to native script
    translated_text: Optional[str] = None  # Full translation to English
    
    def to_dict(self) -> Dict:
        return {
            'primary_language': self.primary_language.value,
            'is_code_mixed': self.is_code_mixed,
            'language_distribution': self.language_distribution,
            'segments': [s.to_dict() for s in self.segments],
            'transliterated': self.transliterated_text,
            'translated': self.translated_text
        }


class CodeMixHandler:
    """
    Handler for Code-Mixed Language Detection and Processing
    
    TECHNICAL NOVELTY:
    - Detects Telugu-English and Hindi-English code-mixing
    - Word-level language identification
    - Maintains technical terms in original language
    - Provides translation assistance
    
    Features:
    - Script-based detection (Devanagari, Telugu script)
    - N-gram based language identification for romanized text
    - Common code-mix pattern recognition
    - Technical glossary preservation
    
    Usage:
        handler = CodeMixHandler()
        result = handler.analyze("Meeting tomorrow ko postpone kar diya")
        
        print(f"Code-mixed: {result.is_code_mixed}")
        print(f"Languages: {result.language_distribution}")
    """
    
    # Common Hindi words (romanized) - frequently used in code-mixing
    HINDI_WORDS = {
        # Verbs
        'karna', 'karo', 'kar', 'kiya', 'karega', 'karenge', 'karte',
        'hona', 'ho', 'hai', 'hain', 'tha', 'the', 'thi', 'hoga', 'honge',
        'dena', 'de', 'do', 'diya', 'denge', 'deta',
        'lena', 'le', 'lo', 'liya', 'lenge', 'leta',
        'bolna', 'bol', 'bolo', 'bola', 'bolte',
        'dekhna', 'dekh', 'dekho', 'dekha', 'dekhte', 'dekhenge',
        'aana', 'aa', 'aao', 'aaya', 'aayenge', 'aate',
        'jana', 'ja', 'jao', 'gaya', 'jayenge', 'jaate',
        'milna', 'mil', 'mila', 'milenge', 'milte',
        'samajhna', 'samajh', 'samjho', 'samjha', 'samajhte',
        'sochna', 'soch', 'socho', 'socha', 'sochte',
        'rakhna', 'rakh', 'rakho', 'rakha', 'rakhte',
        'bhejana', 'bhej', 'bhejo', 'bheja', 'bhejte',
        'banana', 'bana', 'banao', 'banaya', 'banate',
        
        # Pronouns
        'mai', 'main', 'mujhe', 'mera', 'mere', 'meri',
        'tu', 'tum', 'tumhe', 'tumhara', 'tumhare', 'tumhari',
        'aap', 'aapka', 'aapke', 'aapki', 'aapko',
        'wo', 'woh', 'uska', 'uske', 'uski', 'unka', 'unke', 'unki',
        'ye', 'yeh', 'iska', 'iske', 'iski', 'inke', 'inki',
        'hum', 'hume', 'humara', 'humare', 'humari',
        'kya', 'kab', 'kaise', 'kahan', 'kyun', 'kaun', 'kitna', 'kitne',
        
        # Common words
        'aur', 'ya', 'lekin', 'par', 'magar', 'toh', 'to', 'bhi',
        'nahi', 'nahin', 'na', 'mat', 'bilkul',
        'bahut', 'bohot', 'zyada', 'kam', 'thoda', 'thode',
        'abhi', 'ab', 'phir', 'fir', 'baad', 'pehle', 'pahle',
        'kal', 'aaj', 'parso', 'roz', 'hamesha',
        'yahan', 'wahan', 'idhar', 'udhar',
        'sab', 'kuch', 'koi', 'kaafi',
        'achha', 'accha', 'theek', 'thik', 'sahi',
        'zaroor', 'zaruri', 'important',
        'please', 'sorry', 'thank', 'thanks',
        
        # Postpositions
        'ko', 'ka', 'ke', 'ki', 'se', 'me', 'mein', 'pe', 'par', 'tak',
        'ke liye', 'ki taraf', 'ke baare', 'ke saath',
        
        # Question words
        'kya', 'kab', 'kaise', 'kyun', 'kahan', 'kaun', 'konsa',
        
        # Time related
        'kal', 'aaj', 'abhi', 'baad', 'pehle', 'jaldi', 'der',
    }
    
    # Common Telugu words (romanized)
    TELUGU_WORDS = {
        # Verbs
        'cheyali', 'chesthanu', 'chestha', 'cheyandi', 'chestham', 'chesaru',
        'undali', 'undi', 'unnaru', 'untaru', 'unnam',
        'ivvali', 'isthanu', 'icharu', 'ivvandi',
        'randi', 'vasthanu', 'vachanu', 'vastham', 'vacharu',
        'vellali', 'veltanu', 'veldam', 'vellaru',
        'cheppali', 'cheppanu', 'cheppandi', 'chepparu',
        'chudali', 'chusthanu', 'chudandi', 'chusaru',
        'teesukovli', 'teesukuntanu', 'teesukunnaru',
        'pampali', 'pamputhanu', 'pamparu',
        'raayali', 'raasthanu', 'raasaru',
        'adagali', 'aduguthanu', 'adigaru',
        'telusukovali', 'telusukuntanu', 'telisindu',
        'finish', 'complete', 'start',
        
        # Pronouns
        'nenu', 'naku', 'na', 'naaku',
        'nuvvu', 'neeku', 'nee', 'meeru', 'meeku', 'mee',
        'aayana', 'aame', 'vaaru', 'vaallu', 'vallu',
        'idi', 'adi', 'avi', 'ivi',
        'manamu', 'manam', 'mana', 'manaki',
        'evaru', 'emi', 'ento', 'ela', 'ekkada', 'eppudu', 'enduku',
        
        # Common words
        'mariyu', 'leda', 'kani', 'kuda', 'matrame',
        'kaadu', 'ledhu', 'avunu', 'sarele',
        'chala', 'koncham', 'ekkuva', 'takkuva',
        'ippudu', 'appudu', 'tarvata', 'mundu',
        'repu', 'ivala', 'ninna',
        'ikkada', 'akkada',
        'antha', 'konni', 'anni', 'evarina',
        'manchidi', 'sari', 'ok', 'okay',
        'tappakunda', 'avasaram',
        
        # Postpositions
        'ki', 'ku', 'ni', 'tho', 'lo', 'nunchi', 'varaku', 'kosam',
        'gurinchi', 'dwara', 'valla',
        
        # Time related
        'repu', 'ivala', 'ippudu', 'tarvata', 'mundu',
    }
    
    # Technical terms to preserve (don't translate)
    TECHNICAL_TERMS = {
        'api', 'server', 'database', 'frontend', 'backend', 'deploy',
        'commit', 'push', 'pull', 'merge', 'branch', 'git', 'github',
        'bug', 'feature', 'sprint', 'scrum', 'agile', 'jira', 'ticket',
        'meeting', 'standup', 'review', 'demo', 'release', 'deadline',
        'email', 'slack', 'zoom', 'teams', 'call', 'update', 'status',
        'project', 'task', 'issue', 'blocker', 'priority', 'urgent',
        'client', 'customer', 'vendor', 'stakeholder', 'manager',
        'report', 'document', 'presentation', 'slide', 'excel', 'pdf',
        'code', 'test', 'debug', 'error', 'fix', 'patch', 'version',
        'login', 'logout', 'password', 'access', 'permission', 'admin',
        'data', 'file', 'folder', 'download', 'upload', 'share', 'link',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
    }
    
    # Script detection patterns
    DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]+')  # Hindi script
    TELUGU_PATTERN = re.compile(r'[\u0C00-\u0C7F]+')  # Telugu script
    
    def __init__(
        self,
        primary_language: Language = Language.ENGLISH,
        preserve_technical_terms: bool = True
    ):
        """
        Initialize the code-mix handler
        
        Args:
            primary_language: Expected primary language
            preserve_technical_terms: Keep technical terms untranslated
        """
        self.primary_language = primary_language
        self.preserve_technical_terms = preserve_technical_terms
        
        # Build word sets for faster lookup
        self.hindi_words = {w.lower() for w in self.HINDI_WORDS}
        self.telugu_words = {w.lower() for w in self.TELUGU_WORDS}
        self.technical_terms = {t.lower() for t in self.TECHNICAL_TERMS}
    
    def analyze(self, text: str) -> CodeMixResult:
        """
        Analyze text for code-mixing
        
        Args:
            text: Input text (possibly code-mixed)
            
        Returns:
            CodeMixResult with language analysis
        """
        # Check for native scripts first
        has_devanagari = bool(self.DEVANAGARI_PATTERN.search(text))
        has_telugu = bool(self.TELUGU_PATTERN.search(text))
        
        # Tokenize and analyze each word
        words = self._tokenize(text)
        segments = []
        language_counts = {lang.value: 0 for lang in Language if lang != Language.UNKNOWN}
        
        current_segment_words = []
        current_language = None
        current_start = 0
        
        pos = 0
        for word in words:
            word_lower = word.lower()
            word_lang = self._identify_word_language(word_lower, has_devanagari, has_telugu)
            
            # Skip technical terms (treat as English)
            if word_lower in self.technical_terms:
                word_lang = Language.ENGLISH
            
            if current_language is None:
                current_language = word_lang
                current_start = pos
            
            if word_lang == current_language or word_lang == Language.UNKNOWN:
                current_segment_words.append(word)
            else:
                # Save current segment
                if current_segment_words:
                    segment_text = ' '.join(current_segment_words)
                    segments.append(LanguageSegment(
                        text=segment_text,
                        language=current_language,
                        start_pos=current_start,
                        end_pos=pos,
                        confidence=0.8
                    ))
                    language_counts[current_language.value] = language_counts.get(current_language.value, 0) + len(current_segment_words)
                
                # Start new segment
                current_segment_words = [word]
                current_language = word_lang
                current_start = pos
            
            pos += len(word) + 1
        
        # Save final segment
        if current_segment_words:
            segment_text = ' '.join(current_segment_words)
            segments.append(LanguageSegment(
                text=segment_text,
                language=current_language,
                start_pos=current_start,
                end_pos=pos,
                confidence=0.8
            ))
            language_counts[current_language.value] = language_counts.get(current_language.value, 0) + len(current_segment_words)
        
        # Calculate distribution
        total_words = sum(language_counts.values())
        distribution = {}
        if total_words > 0:
            for lang, count in language_counts.items():
                if count > 0:
                    distribution[lang] = round(count / total_words * 100, 1)
        
        # Determine primary language and if code-mixed
        primary = self._determine_primary_language(distribution)
        is_mixed = len([d for d in distribution.values() if d > 10]) > 1
        
        return CodeMixResult(
            original_text=text,
            primary_language=primary,
            segments=segments,
            is_code_mixed=is_mixed,
            language_distribution=distribution
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple whitespace tokenization, preserving punctuation
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _identify_word_language(
        self,
        word: str,
        has_devanagari: bool,
        has_telugu: bool
    ) -> Language:
        """Identify language of a single word"""
        word_lower = word.lower()
        
        # Check for native scripts
        if self.DEVANAGARI_PATTERN.search(word):
            return Language.HINDI
        if self.TELUGU_PATTERN.search(word):
            return Language.TELUGU
        
        # Check word lists for romanized text
        if word_lower in self.hindi_words:
            return Language.HINDI
        if word_lower in self.telugu_words:
            return Language.TELUGU
        
        # Check if it's a common English word or technical term
        if word_lower in self.technical_terms:
            return Language.ENGLISH
        
        # Use character patterns for romanized text
        # Hindi romanized often has: aa, ee, oo, kh, gh, ch, th, dh, bh, ph
        hindi_patterns = ['aa', 'ee', 'oo', 'kh', 'gh', 'ch', 'th', 'dh', 'bh', 'ph', 'sh']
        for pattern in hindi_patterns:
            if pattern in word_lower and len(word) > 3:
                # Could be Hindi, but need more evidence
                if has_devanagari or any(w in self.hindi_words for w in [word_lower[:4], word_lower[-4:]]):
                    return Language.HINDI
        
        # Telugu romanized patterns: ndu, amu, anu, undi, undi
        telugu_patterns = ['ndu', 'amu', 'anu', 'undi', 'undi', 'alu', 'elu']
        for pattern in telugu_patterns:
            if pattern in word_lower and len(word) > 3:
                if has_telugu or any(w in self.telugu_words for w in [word_lower[:4], word_lower[-4:]]):
                    return Language.TELUGU
        
        # Default to English for romanized text
        if word.isascii():
            return Language.ENGLISH
        
        return Language.UNKNOWN
    
    def _determine_primary_language(self, distribution: Dict[str, float]) -> Language:
        """Determine the primary language from distribution"""
        if not distribution:
            return Language.ENGLISH
        
        max_lang = max(distribution.items(), key=lambda x: x[1])
        
        # Check for code-mixed
        en_pct = distribution.get('en', 0)
        hi_pct = distribution.get('hi', 0)
        te_pct = distribution.get('te', 0)
        
        if hi_pct > 20 and en_pct > 20:
            return Language.MIXED_HI_EN
        if te_pct > 20 and en_pct > 20:
            return Language.MIXED_TE_EN
        
        lang_map = {
            'en': Language.ENGLISH,
            'hi': Language.HINDI,
            'te': Language.TELUGU,
        }
        return lang_map.get(max_lang[0], Language.ENGLISH)
    
    def get_english_segments(self, result: CodeMixResult) -> List[str]:
        """Extract only English segments"""
        return [s.text for s in result.segments if s.language == Language.ENGLISH]
    
    def get_hindi_segments(self, result: CodeMixResult) -> List[str]:
        """Extract only Hindi segments"""
        return [s.text for s in result.segments if s.language == Language.HINDI]
    
    def get_telugu_segments(self, result: CodeMixResult) -> List[str]:
        """Extract only Telugu segments"""
        return [s.text for s in result.segments if s.language == Language.TELUGU]
    
    def create_glossary_entry(self, hindi_word: str, english_meaning: str):
        """Add a word to the Hindi glossary"""
        self.hindi_words.add(hindi_word.lower())
    
    def add_technical_term(self, term: str):
        """Add a technical term to preserve"""
        self.technical_terms.add(term.lower())
    
    def format_bilingual(self, result: CodeMixResult) -> str:
        """
        Format code-mixed text with language annotations
        
        Returns text with [HI] [EN] [TE] markers
        """
        parts = []
        for segment in result.segments:
            lang_tag = {
                Language.ENGLISH: "[EN]",
                Language.HINDI: "[HI]",
                Language.TELUGU: "[TE]",
                Language.MIXED_HI_EN: "[HI-EN]",
                Language.MIXED_TE_EN: "[TE-EN]",
            }.get(segment.language, "[?]")
            
            parts.append(f"{lang_tag} {segment.text}")
        
        return ' '.join(parts)


# Common Hindi-English phrases with translations
HINDI_ENGLISH_PHRASES = {
    'kar diya': 'done',
    'ho gaya': 'completed',
    'karna hai': 'need to do',
    'karna padega': 'will have to do',
    'dekhte hain': "let's see",
    'baat karte hain': "let's discuss",
    'samajh gaya': 'understood',
    'theek hai': 'okay',
    'koi problem nahi': 'no problem',
    'time nahi hai': 'no time',
    'kab tak': 'by when',
    'kaise karein': 'how to do',
}

# Common Telugu-English phrases with translations
TELUGU_ENGLISH_PHRASES = {
    'chesthanu': 'I will do',
    'chesaru': 'they did',
    'cheyali': 'need to do',
    'chudandi': 'please see',
    'cheppandi': 'please tell',
    'veltanu': 'I will go',
    'vasthanu': 'I will come',
    'telusukovali': 'need to find out',
    'finish chestha': 'I will finish',
    'complete chesaru': 'they completed',
}
