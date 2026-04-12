"""
Smart Content Analyzer v3 - Content-Aware Question Generation
==============================================================
Improved question generation that:
1. Extracts questions from actual content patterns
2. Uses diverse, context-specific question types
3. Only compares truly related concepts
4. Prioritizes based on content importance
5. Generates unique questions for each input
"""
import re
import math
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import random
from nlp.lang_classifier import get_classifier as _get_lang_clf
from nlp.preprocessor import TextPreprocessor

@dataclass
class ExtractedConcept:
    """A key concept with context"""
    term: str
    definition: str
    frequency: int
    importance_score: float
    context_type: str = "general"  # general, process, comparison, cause_effect, example
    related_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'term': self.term,
            'definition': self.definition,
            'frequency': self.frequency,
            'importance': round(self.importance_score, 2),
            'related': self.related_terms
        }


@dataclass
class GeneratedQuestion:
    """An auto-generated study question"""
    question: str
    answer_hint: str
    question_type: str  # factual, conceptual, analytical, application, synthesis
    difficulty: str     # easy, medium, hard
    source_sentence: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'hint': self.answer_hint,
            'type': self.question_type,
            'difficulty': self.difficulty
        }


@dataclass
class ContentAnalysis:
    """Complete content analysis result"""
    title: str
    executive_summary: str
    key_sentences: List[str]
    concepts: List[ExtractedConcept]
    questions: List[GeneratedQuestion]
    related_topics: List[str]
    
    word_count: int
    sentence_count: int
    reading_time_minutes: float
    
    action_items: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    deadlines: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'executive_summary': self.executive_summary,
            'key_sentences': self.key_sentences,
            'concepts': [c.to_dict() for c in self.concepts],
            'questions': [q.to_dict() for q in self.questions],
            'related_topics': self.related_topics,
            'statistics': {
                'words': self.word_count,
                'sentences': self.sentence_count,
                'reading_time': self.reading_time_minutes
            },
            'meeting_items': {
                'actions': self.action_items,
                'decisions': self.decisions,
                'deadlines': self.deadlines
            }
        }


class SmartAnalyzer:
    """
    Enhanced Content Analyzer with Content-Aware Question Generation
    
    Key improvements:
    - Questions based on actual content patterns (not templates)
    - Diverse question types based on what's in the text
    - Smart comparison (only when concepts are related)
    - Priority-based question ordering
    """
    
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
        'i', 'me', 'my', 'as', 'if', 'then', 'so', 'than', 'such', 'when',
        'where', 'which', 'who', 'what', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'any', 'no', 'not',
        'only', 'own', 'same', 'just', 'also', 'very', 'even', 'back', 'now',
        'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other',
        'use', 'used', 'using', 'make', 'made', 'get', 'got', 'go', 'went',
        'come', 'came', 'take', 'took', 'see', 'saw', 'know', 'knew', 'think',
        'said', 'tell', 'told', 'ask', 'asked', 'well', 'much', 'thing', 'things',
        'like', 'really', 'want', 'way', 'going', 'something', 'actually',
        'yeah', 'yes', 'okay', 'ok', 'um', 'uh', 'basically', 'literally',
    }
    
    # Patterns to detect content types for question generation
    CONTENT_PATTERNS = {
        'cause_effect': [
            r'because\s+(.+?)(?:\.|,)',
            r'leads?\s+to\s+(.+?)(?:\.|,)',
            r'results?\s+in\s+(.+?)(?:\.|,)',
            r'causes?\s+(.+?)(?:\.|,)',
            r'due\s+to\s+(.+?)(?:\.|,)',
            r'therefore\s+(.+?)(?:\.|,)',
            r'consequently\s+(.+?)(?:\.|,)',
        ],
        'process': [
            r'(?:first|step\s*1)[,\s]+(.+?)(?:\.|then)',
            r'process\s+(?:of|for)\s+(.+?)(?:\.|,)',
            r'how\s+to\s+(.+?)(?:\.|,)',
            r'by\s+(?:doing|using|applying)\s+(.+?)(?:\.|,)',
            r'method\s+(?:of|for)\s+(.+?)(?:\.|,)',
        ],
        'comparison': [
            r'(?:unlike|compared\s+to|versus|vs\.?)\s+(.+?)(?:\.|,)',
            r'(?:similar|different)\s+(?:to|from)\s+(.+?)(?:\.|,)',
            r'(?:more|less)\s+\w+\s+than\s+(.+?)(?:\.|,)',
            r'both\s+(.+?)\s+and\s+(.+?)(?:\.|,)',
        ],
        'example': [
            r'(?:for\s+example|for\s+instance|such\s+as)[,\s]+(.+?)(?:\.|$)',
            r'(?:like|including)[,\s]+(.+?)(?:\.|,)',
            r'one\s+example\s+(?:is|of)\s+(.+?)(?:\.|,)',
        ],
        'definition': [
            r'(\w+(?:\s+\w+)*)\s+(?:is|are|refers\s+to|means)\s+(.+?)(?:\.|,)',
            r'(\w+(?:\s+\w+)*)\s+can\s+be\s+defined\s+as\s+(.+?)(?:\.|,)',
        ],
        'importance': [
            r'(?:important|crucial|essential|significant|key)\s+(?:because|for|to)\s+(.+?)(?:\.|,)',
            r'the\s+(?:main|primary|key)\s+(?:reason|purpose|goal)\s+(.+?)(?:\.|,)',
        ],
        'benefit': [
            r'(?:benefit|advantage|helps?|allows?|enables?)\s+(.+?)(?:\.|,)',
            r'(?:can|will)\s+(?:help|improve|enhance)\s+(.+?)(?:\.|,)',
        ],
        'challenge': [
            r'(?:challenge|problem|issue|difficulty|drawback)\s+(?:is|of|with)\s+(.+?)(?:\.|,)',
            r'(?:however|but|although)\s+(.+?)(?:\.|,)',
        ],
    }
    
    TOPIC_KEYWORDS = {
        'Social Media': ['instagram', 'facebook', 'twitter', 'tiktok', 'youtube', 'social', 'post', 'share', 'followers', 'content', 'viral', 'influencer'],
        'Technology': ['software', 'hardware', 'computer', 'digital', 'internet', 'app', 'platform', 'system', 'data', 'algorithm', 'ai', 'machine'],
        'Business': ['company', 'market', 'revenue', 'profit', 'customer', 'sales', 'product', 'service', 'brand', 'strategy'],
        'Education': ['learn', 'student', 'teach', 'school', 'course', 'study', 'knowledge', 'training', 'skill'],
        'Science': ['research', 'experiment', 'theory', 'study', 'discovery', 'scientific', 'hypothesis', 'evidence'],
        'Health': ['health', 'medical', 'disease', 'treatment', 'patient', 'doctor', 'medicine', 'symptom'],
        'Communication': ['message', 'share', 'connect', 'network', 'communicate', 'interact', 'conversation'],
        'Marketing': ['marketing', 'advertising', 'promotion', 'brand', 'campaign', 'audience', 'engagement'],
        'Finance': ['money', 'payment', 'finance', 'bank', 'invest', 'budget', 'cost', 'price'],
        'Computer Science': ['array', 'linked', 'tree', 'graph', 'algorithm', 'complexity', 'node', 'pointer', 'stack', 'queue', 'hash', 'binary', 'sorting', 'searching', 'recursion', 'data', 'structure'],
        'Programming': ['function', 'variable', 'loop', 'class', 'object', 'method', 'code', 'python', 'java', 'program', 'compile', 'runtime', 'memory'],
        'Mathematics': ['equation', 'theorem', 'proof', 'formula', 'calculate', 'probability', 'statistics', 'matrix', 'vector', 'calculus'],
        'Electronics': ['circuit', 'signal', 'frequency', 'voltage', 'current', 'sensor', 'camera', 'audio', 'visual', 'display', 'colour', 'sound'],
        'Cinema & Media': ['cinema', 'film', 'movie', 'audio', 'visual', 'sound', 'colour', 'camera', 'story', 'director', 'playback', 'studio'],
    }
    
    def __init__(self):
        self._tfidf_cache = {}
        self._preprocessor = TextPreprocessor(remove_fillers=True, include_indian_fillers=True)
    
    def analyze(self, text: str, title: str = "Document") -> ContentAnalysis:
        """Analyze text and extract structured content"""
        text = self._preprocessor.clean_for_nlp(text)
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        if not sentences:
            return self._empty_analysis(title)
        
        tfidf_scores = self._calculate_tfidf(sentences)
        
        # Extract content patterns for question generation
        content_patterns = self._extract_content_patterns(clean_text, sentences)
        
        key_sentences = self._extract_key_sentences(sentences, tfidf_scores, max_sentences=5)
        executive_summary = self._generate_summary(key_sentences)
        concepts = self._extract_concepts(clean_text, sentences, tfidf_scores, content_patterns, max_concepts=8)
        
        # Generate content-aware questions
        questions = self._generate_content_aware_questions(
            concepts, sentences, content_patterns, max_questions=8
        )
        
        related_topics = self._find_related_topics(clean_text, max_topics=6)
        
        action_items = self._extract_actions(clean_text)
        decisions = self._extract_decisions(clean_text)
        deadlines = self._extract_deadlines(clean_text)
        
        word_count = len(clean_text.split())
        reading_time = word_count / 200
        
        return ContentAnalysis(
            title=title,
            executive_summary=executive_summary,
            key_sentences=key_sentences,
            concepts=concepts,
            questions=questions,
            related_topics=related_topics,
            word_count=word_count,
            sentence_count=len(sentences),
            reading_time_minutes=round(reading_time, 1),
            action_items=action_items,
            decisions=decisions,
            deadlines=deadlines
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        lines = text.split('\n')
        content_lines = []
        
        skip_patterns = [
            r'^={2,}', r'^-{2,}', r'^\[.*\]$',
            r'^transcript:', r'^audio:', r'^duration:',
            r'^confidence:', r'^words:', r'^timestamps:',
            r'^generated:', r'^echonotes',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
            
            line = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', line).strip()
            
            if line and len(line) > 5:
                content_lines.append(line)
        
        text = ' '.join(content_lines)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences - handles English, Hindi, Telugu, and unpunctuated transcripts"""
        # Hindi/Telugu use Devanagari danda (।) as sentence terminator
        text = text.replace('।', '. ').replace('|', '. ')
        
        # Protect abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s+', r'\1<DOT> ', text)
        
        # Try standard punctuation-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
        sentences = [s for s in sentences if len(s) >= 15 and len(s.split()) >= 3]
        
        # Fallback for unpunctuated text (Vosk transcripts, raw pasted text)
        if len(sentences) <= 2 and len(text.split()) > 50:
            sentences = self._split_unpunctuated(text)
        
        return sentences
    
    def _split_unpunctuated(self, text: str) -> List[str]:
        """Split unpunctuated text into sentence-like segments.
        Handles English, Hindi (Devanagari), and Telugu scripts."""
        words = text.split()
        if len(words) < 10:
            return [text] if len(text) >= 20 else []
        
        # Detect script: check first 200 chars for Devanagari/Telugu
        _lang = _get_lang_clf().predict(text[:500])
        has_devanagari = _lang.label in ('hi', 'hi_en')
        has_telugu     = _lang.label in ('te', 'te_en')
        
        if has_devanagari:
            # Hindi markers: sentence-ending particles, conjunctions, postpositions
            markers = {
                'और', 'लेकिन', 'परंतु', 'किंतु', 'तथा', 'एवं',
                'इसलिए', 'क्योंकि', 'जबकि', 'हालांकि', 'फिर',
                'इसके', 'उसके', 'जिसमें', 'जिससे', 'ताकि',
                'है', 'हैं', 'था', 'थे', 'थी', 'होता', 'करता',
                'इस', 'उस', 'यह', 'वह', 'जो', 'कि',
            }
            min_words, max_words = 10, 30
        elif has_telugu:
            # Telugu markers
            markers = {
                'మరియు', 'కానీ', 'అయితే', 'కాబట్టి', 'ఎందుకంటే',
                'అందువల్ల', 'తర్వాత', 'ముందు', 'ఇది', 'అది',
                'చేస్తుంది', 'ఉంది', 'ఉన్నాయి', 'అవసరం',
                'ద్వారా', 'కోసం', 'లో', 'గా', 'తో',
            }
            min_words, max_words = 8, 25
        else:
            # English markers
            markers = {
                'and', 'but', 'however', 'also', 'then', 'so', 'because',
                'which', 'where', 'when', 'while', 'although', 'meanwhile',
                'moreover', 'furthermore', 'therefore', 'thus', 'hence',
                'the', 'this', 'that', 'these', 'those', 'in', 'on', 'at',
                'after', 'before', 'during', 'following', 'by',
            }
            min_words, max_words = 15, 40
        
        segments = []
        current = []
        for word in words:
            current.append(word)
            # Split when we hit a marker after accumulating enough words
            if len(current) >= min_words and word.lower() in markers:
                seg = ' '.join(current[:-1]).strip()
                if len(seg.split()) >= max(5, min_words // 2):
                    segments.append(seg)
                current = [word]
            # Force split at max_words
            elif len(current) >= max_words:
                seg = ' '.join(current).strip()
                if len(seg.split()) >= max(5, min_words // 2):
                    segments.append(seg)
                current = []
        
        # Remaining words
        if current:
            seg = ' '.join(current).strip()
            if len(seg.split()) >= max(5, min_words // 2):
                segments.append(seg)
        
        return segments if segments else [text[:500]]
    
    def _calculate_tfidf(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate TF-IDF scores"""
        term_doc_freq = Counter()
        term_freq = Counter()
        
        for sent in sentences:
            words = self._tokenize(sent)
            unique_words = set(words)
            for word in unique_words:
                term_doc_freq[word] += 1
            for word in words:
                term_freq[word] += 1
        
        n_docs = len(sentences)
        tfidf = {}
        
        for term, tf in term_freq.items():
            df = term_doc_freq[term]
            idf = math.log((n_docs + 1) / (df + 1)) + 1
            tfidf[term] = tf * idf
        
        if tfidf:
            max_score = max(tfidf.values())
            tfidf = {k: v / max_score for k, v in tfidf.items()}
        
        return tfidf
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in self.STOPWORDS]
    
    def _extract_content_patterns(self, text: str, sentences: List[str]) -> Dict[str, List[Dict]]:
        """Extract content patterns for intelligent question generation"""
        patterns_found = {
            'cause_effect': [],
            'process': [],
            'comparison': [],
            'example': [],
            'definition': [],
            'importance': [],
            'benefit': [],
            'challenge': [],
        }
        
        text_lower = text.lower()
        
        for pattern_type, patterns in self.CONTENT_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        content = ' '.join(match)
                    else:
                        content = match
                    
                    content = content.strip()
                    if len(content) > 10 and len(content) < 300:
                        # Find the source sentence
                        source = ""
                        for sent in sentences:
                            if content[:30].lower() in sent.lower():
                                source = sent
                                break
                        
                        patterns_found[pattern_type].append({
                            'content': content,
                            'source': source
                        })
        
        return patterns_found
    
    def _extract_key_sentences(self, sentences: List[str], tfidf_scores: Dict[str, float], max_sentences: int = 5) -> List[str]:
        """Extract key sentences"""
        if len(sentences) <= max_sentences:
            return sentences
        
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences), tfidf_scores)
            scored.append((i, sent, score))
        
        scored.sort(key=lambda x: -x[2])
        selected = scored[:max_sentences]
        selected.sort(key=lambda x: x[0])
        
        return [sent for _, sent, _ in selected]
    
    def _score_sentence(self, sentence: str, position: int, total: int, tfidf_scores: Dict[str, float]) -> float:
        """Score sentence importance"""
        score = 0.0
        
        # Position score
        rel_pos = position / max(1, total - 1)
        if rel_pos < 0.15:
            score += 0.25
        elif rel_pos > 0.85:
            score += 0.15
        
        # Length score
        words = sentence.split()
        if 12 <= len(words) <= 35:
            score += 0.2
        
        # TF-IDF score
        tokens = self._tokenize(sentence)
        if tokens:
            tfidf_score = sum(tfidf_scores.get(t, 0) for t in tokens) / len(tokens)
            score += tfidf_score * 0.4
        
        # Important phrases
        sent_lower = sentence.lower()
        important_phrases = ['important', 'key', 'main', 'significant', 'essential', 'crucial', 
                           'therefore', 'conclusion', 'result', 'purpose', 'goal']
        for phrase in important_phrases:
            if phrase in sent_lower:
                score += 0.15
                break
        
        return score
    
    def _generate_summary(self, key_sentences: List[str], max_words: int = 60) -> str:
        """Generate concise executive summary from key sentences (max ~60 words)"""
        if not key_sentences:
            return ""
        
        summary_parts = []
        word_count = 0
        
        for sent in key_sentences[:3]:  # Use at most 3 sentences
            words = sent.split()
            if word_count + len(words) <= max_words:
                summary_parts.append(sent.strip())
                word_count += len(words)
            elif word_count == 0:
                # First sentence too long - truncate it
                summary_parts.append(' '.join(words[:max_words]) + '...')
                break
            else:
                break
        
        return ' '.join(summary_parts)
    
    def _extract_concepts(self, text: str, sentences: List[str], tfidf_scores: Dict[str, float], 
                         content_patterns: Dict, max_concepts: int = 8) -> List[ExtractedConcept]:
        """Extract key concepts with context type"""
        concepts = {}
        text_lower = text.lower()
        
        # Words to exclude (common sentence starters, example data indicators, adjectives)
        exclude_words = {
            'note', 'example', 'following', 'below', 'above', 'here', 'there',
            'first', 'second', 'third', 'next', 'last', 'finally', 'also',
            'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
            'elements', 'items', 'values', 'zero', 'one', 'two', 'three',
            'european', 'american', 'asian', 'african', 'western', 'eastern',
            'simple', 'complex', 'basic', 'advanced', 'common', 'different',
            'important', 'useful', 'helpful', 'similar', 'various', 'several',
            'specific', 'particular', 'certain', 'other', 'another', 'same',
            # Generic words that are not meaningful concepts on their own
            'digital', 'small', 'large', 'new', 'many', 'much', 'most',
            'main', 'key', 'major', 'good', 'best', 'better', 'well',
            'high', 'low', 'more', 'less', 'very', 'just', 'only',
            'businesses', 'business', 'systems', 'system', 'tools', 'tool',
            'over', 'under', 'through', 'between', 'within', 'during',
            'bring', 'brings', 'brought', 'watch', 'watching', 'watched',
            'hear', 'heard', 'hearing', 'make', 'makes', 'making', 'made',
            'take', 'takes', 'taking', 'took', 'come', 'comes', 'coming',
            'home', 'trust', 'deliver', 'draw', 'plays', 'begin', 'begins',
            'recreate', 'capture', 'catches', 'preserves', 'elevate',
        }
        
        # Extract words that appear inside examples (quotes, after "For example")
        example_words = set()
        # Find content inside quotes
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        for q in quoted:
            example_words.update(w.lower() for w in re.findall(r'\b([A-Za-z]+)\b', q))
        # Find content inside curly braces (dictionary/set examples)
        braced = re.findall(r'\{([^}]+)\}', text)
        for b in braced:
            example_words.update(w.lower() for w in re.findall(r'\b([A-Za-z]+)\b', b))
        # Find content inside parentheses that looks like examples
        parens = re.findall(r'\(([^)]+)\)', text)
        for p in parens:
            if ',' in p:  # Likely a list of examples
                example_words.update(w.lower() for w in re.findall(r'\b([A-Za-z]+)\b', p))
        
        # METHOD 1: Find terms with explicit definitions (HIGHEST PRIORITY)
        # Patterns like "A tuple is a collection..."
        definition_patterns = [
            r'\bA\s+([a-z]+)\s+is\s+a\s+([^.]+\.)',  # "A tuple is a..."
            r'\bAn?\s+([a-z]+)\s+is\s+an?\s+([^.]+\.)',  # "An X is an..."
            r'\b([A-Z][a-z]+)\s+is\s+a\s+collection\s+([^.]+\.)',  # "Set is a collection..."
            r'\b([A-Z][a-z]+)\s+(?:is|are)\s+(?:used|defined|called)\s+([^.]+\.)',
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).lower()
                definition = match.group(0)  # Full matched sentence
                
                if term not in exclude_words and term not in example_words and len(term) > 2:
                    if term not in concepts:
                        concepts[term] = {
                            'display': term.title(),
                            'score': 2.0,  # High score for defined terms
                            'freq': text_lower.count(term),
                            'definition': definition.strip(),
                            'context_type': 'definition'
                        }
                    else:
                        concepts[term]['score'] += 1.0
                        if not concepts[term]['definition']:
                            concepts[term]['definition'] = definition.strip()
        
        # METHOD 2: Find compound technical terms
        compound_patterns = [
            r'\b(zero[- ]based\s+indexing)\b',
            r'\b(key[:\s]*value\s+pairs?)\b',
            r'\b(duplicate\s+elements?)\b',
            r'\b([a-z]+[- ]based\s+[a-z]+)\b',
            r'\b(data\s+types?)\b',
            r'\b(data\s+structures?)\b',
        ]
        
        for pattern in compound_patterns:
            matches = re.findall(pattern, text_lower)
            for term in matches:
                term = term.strip()
                if term and term not in concepts and len(term) > 5:
                    concepts[term] = {
                        'display': term.title(),
                        'score': 1.5,
                        'freq': text_lower.count(term),
                        'definition': '',
                        'context_type': 'technical'
                    }
        
        # METHOD 3: Technical terms (nouns with definitions nearby)
        # Only add proper nouns that are NOT in example context
        proper_nouns = re.findall(r'\b([A-Z][a-z]+)\b', text)
        for term in proper_nouns:
            term_lower = term.lower()
            
            # Skip if it's excluded, in example data, or a stopword
            if (term_lower in exclude_words or 
                term_lower in example_words or 
                term_lower in self.STOPWORDS or
                len(term) <= 2):
                continue
            
            # Check if this term has a definition nearby
            has_definition = bool(re.search(
                rf'\b{re.escape(term)}\s+(?:is|are|refers to|means)\s+(?:a|an|the)',
                text, re.IGNORECASE
            ))
            
            freq = len(re.findall(r'\b' + re.escape(term_lower) + r'\b', text_lower))
            
            if term_lower not in concepts:
                score = 0.3 * freq
                if has_definition:
                    score += 1.0  # Boost terms with definitions
                
                concepts[term_lower] = {
                    'display': term,
                    'score': score,
                    'freq': freq,
                    'definition': '',
                    'context_type': 'general'
                }
        
        # METHOD 4: High TF-IDF terms (but filter carefully)
        top_terms = sorted(tfidf_scores.items(), key=lambda x: -x[1])[:20]
        for term, score in top_terms:
            if (term not in concepts and 
                term not in exclude_words and 
                term not in example_words and
                len(term) > 3):
                
                freq = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                if freq >= 2:
                    concepts[term] = {
                        'display': term.title(),
                        'score': score * 0.5,  # Lower weight for TF-IDF only terms
                        'freq': freq,
                        'definition': '',
                        'context_type': 'general'
                    }
        
        # Determine context type based on patterns
        for term, data in concepts.items():
            if data['context_type'] == 'general':
                for pattern_type, patterns in content_patterns.items():
                    for p in patterns:
                        if term in p['content'].lower():
                            data['context_type'] = pattern_type
                            break
        
        # Extract definitions for terms that don't have them
        for term, data in concepts.items():
            if not data['definition']:
                definition = self._extract_definition(term, sentences, text)
                if definition:
                    data['definition'] = definition
                    data['score'] += 0.5
        
        # Sort by score (defined terms will be at top)
        sorted_concepts = sorted(concepts.items(), key=lambda x: -x[1]['score'])
        
        result = []
        seen_terms = set()
        
        for term, data in sorted_concepts:
            if len(result) >= max_concepts:
                break
            
            # Skip if similar term already added
            skip = False
            for seen in seen_terms:
                if term in seen or seen in term:
                    skip = True
                    break
            if skip:
                continue
            
            definition = data['definition'] or self._generate_fallback_definition(term, sentences)
            
            result.append(ExtractedConcept(
                term=data['display'],
                definition=definition,
                frequency=data['freq'],
                importance_score=min(1.0, data['score'] / 2.0),  # Normalize score
                context_type=data['context_type']
            ))
            seen_terms.add(term)
        
        # ── Multilingual fallback: if no concepts found (Hindi/Telugu text) ──
        if not result:
            result = self._extract_concepts_multilingual(text, sentences, tfidf_scores, max_concepts)
        
        return result
    
    def _extract_concepts_multilingual(self, text: str, sentences: List[str], 
                                        tfidf_scores: Dict[str, float],
                                        max_concepts: int = 8) -> List[ExtractedConcept]:
        """Extract key concepts from Hindi/Telugu/multilingual text using frequency analysis."""
        # Detect script
        sample = text[:500]
        has_devanagari = any('\u0900' <= c <= '\u097F' for c in sample)
        has_telugu = any('\u0C00' <= c <= '\u0C7F' for c in sample)
        
        # Script-specific stopwords
        hi_stops = {
            'है', 'हैं', 'का', 'के', 'की', 'में', 'से', 'को', 'और', 'पर',
            'ने', 'या', 'एक', 'यह', 'इस', 'जो', 'तो', 'भी', 'था', 'थे', 'थी',
            'कि', 'लिए', 'हो', 'कर', 'साथ', 'वह', 'जा', 'रहे', 'रहा', 'रही',
            'अपने', 'होता', 'करता', 'करते', 'होते', 'किया', 'गया', 'गई', 'अन्य',
            'लेकिन', 'परंतु', 'तथा', 'अगर', 'जैसे', 'बहुत', 'कुछ', 'इसके',
            'उनके', 'उसके', 'जिसमें', 'इसलिए', 'होती', 'करती', 'बाद', 'अब',
            'द्वारा', 'सकता', 'सकते', 'सकती', 'दिया', 'ऐसे', 'उन', 'छोटे',
            'बना', 'बनती', 'बनते', 'माध्यम', 'उठाने', 'पूरा', 'कई', 'अक्सर',
            'बेहतर', 'पारंपरिक', 'सहायक', 'जैसा', 'जैसी', 'करके', 'उपयोग',
            'नया', 'सभी', 'प्रकार', 'बहुत', 'बेहद', 'वहां', 'जहां', 'यहां',
            'रहता', 'रहती', 'रहते', 'करता', 'करती', 'करते', 'होता', 'होती',
            'दिया', 'लिया', 'किया', 'गया', 'गई', 'रहा', 'रही', 'रहे',
            'वाले', 'वाली', 'वालों', 'जिसने', 'उसने', 'इसके', 'उसके',
            'अपनी', 'अपने', 'अपना', 'मेरे', 'मेरा', 'मेरी',
            'सदैव', 'हमेशा', 'कभी', 'कोई', 'दूसरे', 'पहले', 'बाद',
            'आगे', 'पीछे', 'ऊपर', 'नीचे', 'अंदर', 'बाहर',
            'महज', 'सिर्फ', 'काफी', 'बड़ी', 'छोटी', 'ज्यादा',
            'तमाम', 'हुआ', 'हुई', 'हुए', 'पड़ी', 'पड़ा', 'लगा',
            'खुद', 'स्वयं', 'जीवन', 'तरह', 'तरफ', 'ओर', 'बारे',
            'विषय', 'दौरान', 'उत्तर', 'प्रश्न', 'बात', 'लोगों',
            'करने', 'होने', 'देने', 'लेने', 'जाने', 'आने', 'रखने',
            'मिलने', 'बनाने', 'रखा', 'बताने', 'पहुंच',
        }
        te_stops = {
            'మరియు', 'ఈ', 'ఆ', 'ఇది', 'అది', 'లో', 'కు', 'తో', 'గా',
            'ని', 'యొక్క', 'కోసం', 'ద్వారా', 'ఉంది', 'ఉన్నాయి', 'అయితే',
            'కానీ', 'అందువల్ల', 'మాత్రమే', 'కూడా', 'వారు', 'వాటిని', 'ఒక',
            'చేయడం', 'చేస్తుంది', 'చేయవచ్చు', 'అవసరం', 'ఉన్న',
        }
        en_stops = self.STOPWORDS
        
        if has_devanagari:
            stopwords = hi_stops
            min_len = 3
        elif has_telugu:
            stopwords = te_stops
            min_len = 3
        else:
            stopwords = en_stops
            min_len = 4
        
        # Count word frequencies, filtering stopwords
        from collections import Counter
        word_freq = Counter()
        for word in text.split():
            clean = re.sub(r'[।,.!?:;()\[\]{}"\'/\\-]', '', word).strip()
            if clean and len(clean) >= min_len and clean.lower() not in stopwords and clean not in stopwords:
                word_freq[clean] += 1
        
        # Also find bigrams (2-word phrases) for compound concepts
        words = text.split()
        for i in range(len(words) - 1):
            w1 = re.sub(r'[।,.!?:;()\[\]{}"\'/\\-]', '', words[i]).strip()
            w2 = re.sub(r'[।,.!?:;()\[\]{}"\'/\\-]', '', words[i+1]).strip()
            if (w1 and w2 and len(w1) >= min_len and len(w2) >= min_len
                and w1.lower() not in stopwords and w1 not in stopwords
                and w2.lower() not in stopwords and w2 not in stopwords):
                bigram = f"{w1} {w2}"
                word_freq[bigram] += 1
        
        # Select top concepts by frequency (require 2+ occurrences for single words)
        result = []
        seen_words = set()
        for term, freq in word_freq.most_common(30):
            if freq < 2 and ' ' not in term:
                continue
            # Skip if a similar term already added
            term_lower = term.lower()
            if any(term_lower in s or s in term_lower for s in seen_words):
                continue
            
            # Find a context sentence for this term
            context = ""
            for sent in sentences:
                if term in sent or term.lower() in sent.lower():
                    # Use max 20 words of context
                    ctx_words = sent.split()[:20]
                    context = ' '.join(ctx_words)
                    if len(ctx_words) == 20:
                        context += '...'
                    break
            
            result.append(ExtractedConcept(
                term=term,
                definition=context,
                frequency=freq,
                importance_score=min(1.0, freq / 10.0),
                context_type='keyword'
            ))
            seen_words.add(term_lower)
            
            if len(result) >= max_concepts:
                break
        
        return result
    
    def _extract_definition(self, term: str, sentences: List[str], text: str) -> str:
        """Extract definition for a term"""
        term_escaped = re.escape(term)
        
        # Definition patterns
        patterns = [
            rf'{term_escaped}\s+(?:is|are|refers to|means)\s+(.+?)(?:\.|,\s+(?:which|that))',
            rf'{term_escaped}\s+can be defined as\s+(.+?)(?:\.|,)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                definition = match.group(1).strip()
                if 10 < len(definition) < 300:
                    return definition[0].upper() + definition[1:] + ('.' if not definition.endswith('.') else '')
        
        # Find best sentence containing term
        for sent in sentences:
            if term in sent.lower():
                sent_lower = sent.lower()
                if any(p in sent_lower for p in ['is', 'are', 'means', 'refers']):
                    if len(sent) > 250:
                        sent = sent[:250].rsplit(' ', 1)[0] + '...'
                    return sent
        
        return ""
    
    def _generate_fallback_definition(self, term: str, sentences: List[str]) -> str:
        """Generate fallback definition"""
        for sent in sentences:
            if term in sent.lower():
                if len(sent) > 250:
                    sent = sent[:250].rsplit(' ', 1)[0] + '...'
                return sent
        return f"A key concept discussed in this content."
    
    def _generate_content_aware_questions(
        self,
        concepts: List[ExtractedConcept],
        sentences: List[str],
        content_patterns: Dict[str, List[Dict]],
        max_questions: int = 8
    ) -> List[GeneratedQuestion]:
        """
        Generate questions based on actual content patterns.
        Supports English, Hindi, and Telugu.
        """
        questions = []
        used_concepts = set()
        
        # Detect language from sentences
        sample = ' '.join(sentences[:3]) if sentences else ''
        is_hindi = any('\u0900' <= c <= '\u097F' for c in sample)
        is_telugu = any('\u0C00' <= c <= '\u0C7F' for c in sample)
        
        # 1. Generate cause-effect questions if pattern exists
        if content_patterns['cause_effect']:
            pattern = content_patterns['cause_effect'][0]
            questions.append(GeneratedQuestion(
                question=f"What causes or leads to the effects described in the content?",
                answer_hint=pattern['source'] if pattern['source'] else pattern['content'],
                question_type='analytical',
                difficulty='medium',
                source_sentence=pattern['source']
            ))
        
        # 2. Generate process/how questions if pattern exists
        if content_patterns['process']:
            pattern = content_patterns['process'][0]
            questions.append(GeneratedQuestion(
                question="Describe the process or steps explained in the content.",
                answer_hint=pattern['source'] if pattern['source'] else pattern['content'],
                question_type='conceptual',
                difficulty='medium',
                source_sentence=pattern['source']
            ))
        
        # 3. Generate example-based question if examples exist
        if content_patterns['example']:
            pattern = content_patterns['example'][0]
            questions.append(GeneratedQuestion(
                question="What examples are provided to illustrate the main concepts?",
                answer_hint=pattern['source'] if pattern['source'] else pattern['content'],
                question_type='factual',
                difficulty='easy',
                source_sentence=pattern['source']
            ))
        
        # 4. Generate importance/purpose question if pattern exists
        if content_patterns['importance']:
            pattern = content_patterns['importance'][0]
            questions.append(GeneratedQuestion(
                question="Why is this topic important or significant?",
                answer_hint=pattern['source'] if pattern['source'] else pattern['content'],
                question_type='conceptual',
                difficulty='medium',
                source_sentence=pattern['source']
            ))
        
        # 5. Generate benefit question if pattern exists
        if content_patterns['benefit']:
            pattern = content_patterns['benefit'][0]
            questions.append(GeneratedQuestion(
                question="What are the main benefits or advantages discussed?",
                answer_hint=pattern['source'] if pattern['source'] else pattern['content'],
                question_type='factual',
                difficulty='easy',
                source_sentence=pattern['source']
            ))
        
        # 6. Generate challenge/problem question if pattern exists
        if content_patterns['challenge']:
            pattern = content_patterns['challenge'][0]
            questions.append(GeneratedQuestion(
                question="What challenges or problems are mentioned in the content?",
                answer_hint=pattern['source'] if pattern['source'] else pattern['content'],
                question_type='analytical',
                difficulty='medium',
                source_sentence=pattern['source']
            ))
        
        # 7. Generate concept-specific questions for remaining slots
        question_counter = 0
        for concept in concepts:
            if len(questions) >= max_questions:
                break
            
            if concept.term in used_concepts:
                continue
            
            used_concepts.add(concept.term)
            question_counter += 1
            
            # Choose question type based on context and counter for variety
            if concept.context_type == 'process':
                question = f"Explain the process or mechanism of {concept.term}."
                q_type = 'conceptual'
                difficulty = 'medium'
            elif concept.context_type == 'cause_effect':
                question = f"What factors influence {concept.term} and what are its effects?"
                q_type = 'analytical'
                difficulty = 'hard'
            elif concept.context_type == 'example':
                question = f"Describe a practical example of {concept.term} and why it's relevant."
                q_type = 'application'
                difficulty = 'medium'
            else:
                # Use varied question starters - rotate based on counter
                # Detect language for question templates
                _sample = concept.definition[:100] if concept.definition else concept.term
                _is_hi = any('\u0900' <= c <= '\u097F' for c in _sample)
                _is_te = any('\u0C00' <= c <= '\u0C7F' for c in _sample)
                
                if _is_hi:
                    starters = [
                        (f"{concept.term} को परिभाषित करें और इसकी मुख्य विशेषताएं बताएं।", 'factual', 'easy'),
                        (f"{concept.term} का उद्देश्य या कार्य क्या है?", 'conceptual', 'medium'),
                        (f"{concept.term} समग्र विषय में कैसे योगदान देता है?", 'analytical', 'medium'),
                        (f"{concept.term} को अपने शब्दों में एक उदाहरण सहित समझाएं।", 'application', 'medium'),
                        (f"यदि {concept.term} अनुपस्थित या भिन्न होता तो क्या होता?", 'analytical', 'hard'),
                        (f"{concept.term} के मुख्य बिंदुओं का सारांश दें।", 'factual', 'easy'),
                    ]
                elif _is_te:
                    starters = [
                        (f"{concept.term} ను నిర్వచించి దాని ముఖ్య లక్షణాలను వివరించండి.", 'factual', 'easy'),
                        (f"{concept.term} యొక్క ఉద్దేశ్యం లేదా పని ఏమిటి?", 'conceptual', 'medium'),
                        (f"{concept.term} మొత్తం అంశానికి ఎలా దోహదం చేస్తుంది?", 'analytical', 'medium'),
                        (f"{concept.term} ను మీ సొంత మాటల్లో ఒక ఉదాహరణతో వివరించండి.", 'application', 'medium'),
                        (f"{concept.term} లేకపోతే లేదా భిన్నంగా ఉంటే ఏమి జరుగుతుంది?", 'analytical', 'hard'),
                        (f"{concept.term} గురించి ముఖ్యమైన అంశాలను సంక్షిప్తంగా చెప్పండి.", 'factual', 'easy'),
                    ]
                else:
                    starters = [
                        (f"Define {concept.term} and explain its key characteristics.", 'factual', 'easy'),
                        (f"What is the purpose or function of {concept.term}?", 'conceptual', 'medium'),
                        (f"How does {concept.term} contribute to the overall topic?", 'analytical', 'medium'),
                        (f"Explain {concept.term} in your own words with an example.", 'application', 'medium'),
                        (f"What would happen if {concept.term} was absent or different?", 'analytical', 'hard'),
                        (f"Summarize the key points about {concept.term}.", 'factual', 'easy'),
                    ]
                q_data = starters[question_counter % len(starters)]
                question = q_data[0]
                q_type = q_data[1]
                difficulty = q_data[2]
            
            questions.append(GeneratedQuestion(
                question=question,
                answer_hint=self._truncate_hint(concept.definition),
                question_type=q_type,
                difficulty=difficulty,
                source_sentence=concept.definition
            ))
        
        # 8. Only add comparison if concepts are actually related (mentioned together)
        if len(concepts) >= 2 and content_patterns['comparison']:
            pattern = content_patterns['comparison'][0]
            # Create structured comparison question
            questions.append(GeneratedQuestion(
                question="Identify and explain the key differences and similarities mentioned in the content.",
                answer_hint=f"Look for contrasting points: {pattern['source'][:100]}..." if pattern['source'] else pattern['content'],
                question_type='analytical',
                difficulty='hard',
                source_sentence=pattern['source']
            ))
        elif len(concepts) >= 2:
            # Check if two concepts appear in the same sentence (truly related)
            c1, c2 = concepts[0], concepts[1]
            for sent in sentences:
                sent_lower = sent.lower()
                if c1.term.lower() in sent_lower and c2.term.lower() in sent_lower:
                    questions.append(GeneratedQuestion(
                        question=f"Compare {c1.term} and {c2.term}. What are the key differences and what do they have in common?",
                        answer_hint=f"Consider: {sent[:100]}..." if len(sent) > 100 else sent,
                        question_type='analytical',
                        difficulty='hard',
                        source_sentence=sent
                    ))
                    break
        
        # 9. Add pros/cons question if both benefit and challenge patterns exist
        if content_patterns['benefit'] and content_patterns['challenge']:
            benefit_hint = content_patterns['benefit'][0]['source'][:80] if content_patterns['benefit'][0]['source'] else ""
            challenge_hint = content_patterns['challenge'][0]['source'][:80] if content_patterns['challenge'][0]['source'] else ""
            questions.append(GeneratedQuestion(
                question="List the advantages and disadvantages discussed. What are the trade-offs?",
                answer_hint=f"Advantages: {benefit_hint}... Disadvantages: {challenge_hint}...",
                question_type='analytical',
                difficulty='hard',
                source_sentence=benefit_hint + " " + challenge_hint
            ))
        
        # 10. Add application question for first concept
        if concepts:
            main_concept = concepts[0]
            questions.append(GeneratedQuestion(
                question=f"How could you apply the concept of {main_concept.term} in a real-world situation?",
                answer_hint=f"Consider practical applications based on: {main_concept.definition[:80]}...",
                question_type='application',
                difficulty='hard',
                source_sentence=main_concept.definition
            ))
        
        # Deduplicate and limit
        seen_questions = set()
        unique_questions = []
        for q in questions:
            q_key = q.question.lower()[:50]
            if q_key not in seen_questions:
                seen_questions.add(q_key)
                unique_questions.append(q)
        
        return unique_questions[:max_questions]
    
    def _truncate_hint(self, text: str, max_length: int = 100) -> str:
        """Truncate hint to max ~20 words for concise display"""
        if not text:
            return ""
        words = text.split()[:20]
        result = ' '.join(words)
        if len(words) == 20:
            result += '...'
        return result
    def _find_related_topics(self, text: str, max_topics: int = 6) -> List[str]:
        """Find related topics - multilingual support"""
        text_lower = text.lower()
        
        # Detect script
        _lang = _get_lang_clf().predict(text[:300])
        has_devanagari = _lang.label in ('hi', 'hi_en')
        has_telugu     = _lang.label in ('te', 'te_en')
        
        if has_devanagari:
            topic_kw = {
                'प्रौद्योगिकी (Technology)': ['डिजिटल', 'तकनीक', 'सॉफ्टवेयर', 'इंटरनेट', 'ऐप', 'कंप्यूटर', 'प्रौद्योगिकी', 'ऑनलाइन'],
                'व्यापार (Business)': ['व्यवसाय', 'व्यापार', 'कंपनी', 'बाजार', 'ग्राहक', 'बिक्री', 'उत्पाद', 'सेवा'],
                'शिक्षा (Education)': ['शिक्षा', 'विद्यार्थी', 'पढ़ाई', 'स्कूल', 'कॉलेज', 'प्रशिक्षण', 'ज्ञान', 'साक्षरता'],
                'स्वास्थ्य (Health)': ['स्वास्थ्य', 'चिकित्सा', 'चिकित्सक', 'रोग', 'उपचार', 'डॉक्टर', 'अस्पताल', 'दवा', 'बीमारी', 'निरोगी', 'औषधीय', 'आरोग्य', 'मृत्यु', 'निमोनिया', 'मरीज'],
                'वित्त (Finance)': ['भुगतान', 'वित्तीय', 'बैंक', 'निवेश', 'लागत', 'मूल्य', 'पैसा', 'ऋण'],
                'कृषि (Agriculture)': ['कृषि', 'खेती', 'फसल', 'किसान', 'सिंचाई', 'मिट्टी', 'उर्वरक'],
                'सुरक्षा (Security)': ['सुरक्षा', 'साइबर', 'हैकर', 'खतरा', 'गोपनीयता', 'संरक्षण'],
                'सरकार (Government)': ['सरकार', 'सरकारी', 'योजना', 'नीति', 'कानून', 'प्रशासन'],
                'पर्यावरण (Environment)': ['पर्यावरण', 'प्रदूषण', 'जलवायु', 'ऊर्जा', 'वन', 'जल'],
                'सामाजिक (Social)': ['समाज', 'सामाजिक', 'संस्कृति', 'महिला', 'युवा', 'परिवार'],
            }
        elif has_telugu:
            topic_kw = {
                'సాంకేతికత (Technology)': ['డిజిటల్', 'సాంకేతికత', 'సాఫ్ట్‌వేర్', 'ఇంటర్నెట్', 'యాప్', 'ఆన్‌లైన్'],
                'వ్యాపారం (Business)': ['వ్యాపారం', 'వ్యవసాయం', 'కంపెనీ', 'మార్కెట్', 'వినియోగదారు', 'ఉత్పత్తి'],
                'విద్య (Education)': ['విద్య', 'విద్యార్థి', 'పాఠశాల', 'శిక్షణ', 'నైపుణ్యం', 'అభ్యాసం'],
                'ఆరోగ్యం (Health)': ['ఆరోగ్యం', 'వైద్యం', 'రోగం', 'చికిత్స', 'డాక్టర్', 'ఆసుపత్రి'],
                'ఆర్థికం (Finance)': ['చెల్లింపు', 'ఆర్థిక', 'బ్యాంకు', 'పెట్టుబడి', 'ధర', 'వ్యయం'],
                'వ్యవసాయం (Agriculture)': ['వ్యవసాయం', 'పంట', 'రైతు', 'నీటి', 'సాగు', 'ఎరువు', 'దిగుబడి'],
                'భద్రత (Security)': ['భద్రత', 'సైబర్', 'హ్యాకర్', 'రక్షణ', 'గోప్యత'],
                'ప్రభుత్వం (Government)': ['ప్రభుత్వ', 'పథకం', 'విధానం', 'చట్టం', 'పాలన'],
                'పర్యావరణం (Environment)': ['పర్యావరణం', 'కాలుష్యం', 'వాతావరణం', 'శక్తి', 'జలం'],
            }
        else:
            topic_kw = self.TOPIC_KEYWORDS
        
        topic_scores = {}
        for topic, keywords in topic_kw.items():
            score = 0
            matched = 0
            for kw in keywords:
                if has_devanagari or has_telugu:
                    # Simple substring match for Indic scripts (no word boundaries)
                    count = text.count(kw)
                else:
                    count = len(re.findall(r'\b' + re.escape(kw) + r'\b', text_lower))
                if count > 0:
                    score += min(count, 5)
                    matched += 1
            
            if matched >= 1 and (has_devanagari or has_telugu):
                topic_scores[topic] = score + matched  # Boost for any match in Indic
            elif matched >= 2:
                topic_scores[topic] = score
        
        sorted_topics = sorted(topic_scores.items(), key=lambda x: -x[1])
        return [topic for topic, _ in sorted_topics[:max_topics]]
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract action items"""
        patterns = [
            r'(?:we|I|you|they)\s+(?:will|should|need to|must|have to)\s+([^.!?]+)',
            r'action\s*(?:item)?[:\s]+([^.!?]+)',
            r'(?:please|kindly)\s+([^.!?]+)',
        ]
        
        actions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                action = match.strip()
                if 15 < len(action) < 200:
                    actions.append(action[0].upper() + action[1:])
        
        return list(set(actions))[:5]
    
    def _extract_decisions(self, text: str) -> List[str]:
        """Extract decisions"""
        patterns = [
            r'(?:we|they)\s+decided\s+(?:to\s+)?([^.!?]+)',
            r'decision[:\s]+([^.!?]+)',
            r'agreed\s+(?:to|on|that)\s+([^.!?]+)',
        ]
        
        decisions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 10 < len(match) < 200:
                    decisions.append(match.strip())
        
        return list(set(decisions))[:5]
    
    def _extract_deadlines(self, text: str) -> List[str]:
        """Extract deadlines"""
        patterns = [
            r'(?:by|before|until)\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)[^.!?]*)',
            r'(?:by|before|until)\s+(tomorrow|today|end of (?:day|week|month))',
            r'deadline[:\s]+([^.!?]+)',
        ]
        
        deadlines = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if 3 < len(match) < 100:
                    deadlines.append(match.strip())
        
        return list(set(deadlines))[:5]
    
    def _empty_analysis(self, title: str) -> ContentAnalysis:
        """Return empty analysis"""
        return ContentAnalysis(
            title=title,
            executive_summary="No content available for analysis.",
            key_sentences=[],
            concepts=[],
            questions=[],
            related_topics=[],
            word_count=0,
            sentence_count=0,
            reading_time_minutes=0.0
        )
