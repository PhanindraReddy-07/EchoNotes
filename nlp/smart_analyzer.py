"""
Smart Content Analyzer - Improved NLP Pipeline
===============================================
Better extraction of:
- Executive Summary (using TF-IDF + position scoring)
- Key Sentences (TextRank improved)
- Key Concepts (noun phrase extraction + frequency)
- Study Questions (template-based from concepts)
- Related Topics (semantic similarity)

Works for BOTH meetings AND lectures/general content.
"""
import re
import math
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class ExtractedConcept:
    """A key concept with context"""
    term: str
    definition: str
    frequency: int
    importance_score: float
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
    question_type: str  # factual, conceptual, analytical
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
    
    # Statistics
    word_count: int
    sentence_count: int
    reading_time_minutes: float
    
    # For meetings
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
    Improved Content Analyzer with better NLP
    
    Uses:
    - TF-IDF for term importance
    - TextRank for sentence extraction
    - Noun phrase patterns for concepts
    - Template-based question generation
    - **ML Model for sentence importance** (custom trained)
    """
    
    # Flag for ML model availability
    _ml_model = None
    _ml_model_loaded = False
    
    # Expanded stopwords
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
        'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next',
        'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able',
        'about', 'after', 'again', 'against', 'because', 'before', 'below',
        'between', 'during', 'into', 'through', 'under', 'until', 'while',
        'above', 'across', 'along', 'among', 'around', 'behind', 'beside',
        'use', 'used', 'using', 'make', 'made', 'get', 'got', 'go', 'went',
        'come', 'came', 'take', 'took', 'see', 'saw', 'know', 'knew', 'think',
        'thought', 'want', 'say', 'said', 'tell', 'told', 'ask', 'asked',
        'work', 'seem', 'feel', 'try', 'leave', 'call', 'keep', 'let', 'begin',
        'show', 'hear', 'play', 'run', 'move', 'live', 'believe', 'bring',
        'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet',
        'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
        'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add',
        'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love',
        'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect',
        'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest',
        'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull',
    }
    
    # Patterns for noun phrases (key concepts)
    NOUN_PHRASE_PATTERNS = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Proper noun phrases
        r'\b([a-z]+(?:\s+[a-z]+){1,2})\s+(?:is|are|was|were)\b',  # X is/are
        r'\b(?:the|a|an)\s+([a-z]+(?:\s+[a-z]+){0,2})\b',  # the/a X
        r'\b([a-z]+ing)\s+([a-z]+)\b',  # -ing noun
        r'\b([a-z]+tion|[a-z]+ment|[a-z]+ness|[a-z]+ity)\b',  # Abstract nouns
    ]
    
    # Question templates
    QUESTION_TEMPLATES = {
        'what_is': [
            "What is {concept}?",
            "Define {concept} in your own words.",
            "Explain what {concept} means.",
        ],
        'how_does': [
            "How does {concept} work?",
            "Describe the process of {concept}.",
            "Explain the mechanism behind {concept}.",
        ],
        'why': [
            "Why is {concept} important?",
            "What is the significance of {concept}?",
            "Why do we need {concept}?",
        ],
        'compare': [
            "Compare {concept1} and {concept2}.",
            "What are the differences between {concept1} and {concept2}?",
            "How does {concept1} relate to {concept2}?",
        ],
        'application': [
            "Give an example of {concept} in practice.",
            "How is {concept} applied in real-world scenarios?",
            "Describe a situation where {concept} would be useful.",
        ],
        'analysis': [
            "What are the advantages and disadvantages of {concept}?",
            "Analyze the impact of {concept}.",
            "What are the key features of {concept}?",
        ],
    }
    
    # Topic keywords for related topics
    TOPIC_KEYWORDS = {
        'Social Media': ['instagram', 'facebook', 'twitter', 'tiktok', 'youtube', 'social', 'post', 'share', 'followers', 'likes', 'content', 'viral', 'influencer'],
        'Technology': ['software', 'hardware', 'computer', 'digital', 'internet', 'app', 'platform', 'system', 'data', 'algorithm', 'ai', 'machine'],
        'Business': ['company', 'market', 'revenue', 'profit', 'customer', 'sales', 'product', 'service', 'brand', 'strategy'],
        'Education': ['learn', 'student', 'teach', 'school', 'course', 'study', 'knowledge', 'training', 'skill'],
        'Science': ['research', 'experiment', 'theory', 'study', 'discovery', 'scientific', 'hypothesis', 'evidence'],
        'Health': ['health', 'medical', 'disease', 'treatment', 'patient', 'doctor', 'medicine', 'symptom'],
        'Communication': ['message', 'share', 'connect', 'network', 'communicate', 'interact', 'conversation'],
        'Media': ['photo', 'video', 'image', 'content', 'media', 'upload', 'download', 'stream'],
        'Marketing': ['marketing', 'advertising', 'promotion', 'brand', 'campaign', 'audience', 'engagement'],
        'User Experience': ['user', 'interface', 'feature', 'design', 'experience', 'usability', 'interaction'],
    }
    
    def __init__(self, use_ml_model: bool = True):
        """
        Initialize SmartAnalyzer.
        
        Args:
            use_ml_model: Whether to use ML model for sentence scoring
        """
        self.use_ml_model = use_ml_model
        self._load_ml_model()
    
    def _load_ml_model(self):
        """Load ML model if available"""
        if not self.use_ml_model:
            return
        
        if SmartAnalyzer._ml_model_loaded:
            return
        
        try:
            from ml.sentence_classifier import SentenceImportanceClassifier
            
            classifier = SentenceImportanceClassifier()
            model_path = classifier.MODEL_DIR / classifier.DEFAULT_MODEL
            
            if model_path.exists():
                classifier.load_model()
                SmartAnalyzer._ml_model = classifier
                SmartAnalyzer._ml_model_loaded = True
                print("[SmartAnalyzer] ML model loaded successfully")
            else:
                print("[SmartAnalyzer] ML model not found. Run 'python train_model.py' to train.")
                SmartAnalyzer._ml_model_loaded = True  # Don't retry
        except ImportError:
            print("[SmartAnalyzer] ML module not available")
            SmartAnalyzer._ml_model_loaded = True
        except Exception as e:
            print(f"[SmartAnalyzer] ML model error: {e}")
            SmartAnalyzer._ml_model_loaded = True
    
    def analyze(self, text: str, title: str = "Document") -> ContentAnalysis:
        """
        Perform comprehensive content analysis
        
        Args:
            text: Input text
            title: Document title
            
        Returns:
            ContentAnalysis with all extracted information
        """
        # Clean and preprocess
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        words = self._tokenize(clean_text)
        
        # Calculate TF-IDF scores
        tfidf_scores = self._calculate_tfidf(sentences)
        
        # Extract components (pass clean_text as document for ML scoring)
        key_sentences = self._extract_key_sentences(sentences, tfidf_scores, num=5, document=clean_text)
        executive_summary = self._generate_summary(sentences, tfidf_scores)
        concepts = self._extract_concepts(clean_text, sentences, tfidf_scores)
        questions = self._generate_questions(concepts, sentences)
        related_topics = self._find_related_topics(clean_text)
        
        # Meeting-specific extraction
        action_items = self._extract_actions(clean_text)
        decisions = self._extract_decisions(clean_text)
        deadlines = self._extract_deadlines(clean_text)
        
        # Statistics
        word_count = len(words)
        sentence_count = len(sentences)
        reading_time = word_count / 200  # ~200 words per minute
        
        return ContentAnalysis(
            title=title,
            executive_summary=executive_summary,
            key_sentences=key_sentences,
            concepts=concepts,
            questions=questions,
            related_topics=related_topics,
            word_count=word_count,
            sentence_count=sentence_count,
            reading_time_minutes=round(reading_time, 1),
            action_items=action_items,
            decisions=decisions,
            deadlines=deadlines
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text - remove transcript metadata"""
        lines = text.split('\n')
        clean_lines = []
        
        # Skip metadata patterns
        skip_patterns = [
            r'^EchoNotes\s+Transcript',
            r'^Audio:',
            r'^Duration:',
            r'^Words:',
            r'^Confidence:',
            r'^TRANSCRIPT:',
            r'^WITH\s+TIMESTAMPS:',
            r'^Recording',
            r'^Generated:',
            r'^={3,}',
            r'^-{3,}',
            r'^\[\d{2}:\d{2}\]',
        ]
        
        import re
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any skip pattern
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            
            if skip:
                continue
            
            # Remove timestamp markers like [00:10]
            line = re.sub(r'\[\d{2}:\d{2}\]', '', line).strip()
            
            # Skip very short lines or metadata-like content
            if len(line) < 10:
                continue
            if line.lower().startswith(('audio', 'duration', 'words', 'confidence', 'transcript')):
                continue
            
            clean_lines.append(line)
        
        text = ' '.join(clean_lines)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix sentence boundaries
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Better sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15:  # Minimum sentence length
                # Capitalize first letter
                if sent and sent[0].islower():
                    sent = sent[0].upper() + sent[1:]
                # Ensure ends with punctuation
                if sent and sent[-1] not in '.!?':
                    sent += '.'
                result.append(sent)
        
        return result
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return [w for w in words if w not in self.STOPWORDS]
    
    def _calculate_tfidf(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate TF-IDF scores for terms"""
        if not sentences:
            return {}
        
        # Document frequency
        doc_freq = Counter()
        term_freq = Counter()
        
        for sent in sentences:
            words = set(self._tokenize(sent))
            for word in words:
                doc_freq[word] += 1
            term_freq.update(self._tokenize(sent))
        
        # Calculate TF-IDF
        num_docs = len(sentences)
        tfidf = {}
        
        for word, tf in term_freq.items():
            df = doc_freq[word]
            idf = math.log(num_docs / (df + 1)) + 1
            tfidf[word] = tf * idf
        
        # Normalize
        max_score = max(tfidf.values()) if tfidf else 1
        return {k: v / max_score for k, v in tfidf.items()}
    
    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total: int,
        tfidf: Dict[str, float],
        document: str = ""
    ) -> float:
        """
        Score a sentence's importance using hybrid approach:
        - ML model score (if available) - 50% weight
        - Rule-based score - 50% weight
        """
        # Try ML model first
        ml_score = None
        if SmartAnalyzer._ml_model is not None and document:
            try:
                ml_score = SmartAnalyzer._ml_model.predict(
                    sentence, document, position, total
                )
            except Exception as e:
                pass  # Fall back to rule-based
        
        # Rule-based scoring
        rule_score = self._rule_based_score(sentence, position, total, tfidf)
        
        # Combine scores
        if ml_score is not None:
            # Hybrid: 50% ML + 50% rule-based
            final_score = 0.5 * ml_score + 0.5 * rule_score
        else:
            final_score = rule_score
        
        return min(final_score, 1.0)
    
    def _rule_based_score(
        self,
        sentence: str,
        position: int,
        total: int,
        tfidf: Dict[str, float]
    ) -> float:
        """Original rule-based sentence scoring"""
        score = 0.0
        words = self._tokenize(sentence)
        
        if not words:
            return 0.0
        
        # TF-IDF score (average of word scores)
        word_scores = [tfidf.get(w, 0) for w in words]
        if word_scores:
            score += sum(word_scores) / len(word_scores) * 0.4
        
        # Position score (first sentences more important)
        if position == 0:
            score += 0.25
        elif position < total * 0.2:
            score += 0.15
        elif position > total * 0.8:
            score += 0.05
        
        # Length score (prefer medium length)
        if 15 <= len(words) <= 35:
            score += 0.15
        elif 10 <= len(words) <= 40:
            score += 0.1
        
        # Keyword indicators
        indicators = [
            'important', 'key', 'main', 'significant', 'essential',
            'allows', 'enables', 'provides', 'offers', 'includes',
            'means', 'defined', 'known', 'called', 'refers',
            'first', 'originally', 'founded', 'created', 'launched',
            'million', 'billion', 'percent', 'over', 'more than'
        ]
        sent_lower = sentence.lower()
        for word in indicators:
            if word in sent_lower:
                score += 0.05
        
        return min(score, 1.0)
    
    def _extract_key_sentences(
        self,
        sentences: List[str],
        tfidf: Dict[str, float],
        num: int = 5,
        document: str = ""
    ) -> List[str]:
        """Extract most important sentences using hybrid ML + rule-based scoring"""
        if len(sentences) <= num:
            return sentences
        
        # Score all sentences
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences), tfidf, document)
            scored.append((score, i, sent))
        
        # Get top sentences by score
        scored.sort(reverse=True)
        top_indices = sorted([idx for _, idx, _ in scored[:num]])
        
        # Return in original order
        return [sentences[i] for i in top_indices]
    
    def _generate_summary(
        self,
        sentences: List[str],
        tfidf: Dict[str, float],
        max_sentences: int = 3
    ) -> str:
        """Generate executive summary"""
        if not sentences:
            return ""
        
        if len(sentences) <= max_sentences:
            return ' '.join(sentences)
        
        # Get top sentences
        key = self._extract_key_sentences(sentences, tfidf, max_sentences)
        return ' '.join(key)
    
    def _extract_concepts(
        self,
        text: str,
        sentences: List[str],
        tfidf: Dict[str, float],
        max_concepts: int = 8
    ) -> List[ExtractedConcept]:
        """Extract key concepts using multiple methods"""
        concepts = {}
        
        # Terms to skip (metadata, common words)
        skip_terms = {
            'transcript', 'audio', 'duration', 'words', 'confidence', 'recording',
            'echonotes', 'timestamps', 'generated', 'notes', 'document',
            'the', 'this', 'that', 'with', 'from', 'have', 'will', 'been',
            'american', 'located', 'define', 'term'
        }
        
        # Method 1: High TF-IDF single words
        sorted_terms = sorted(tfidf.items(), key=lambda x: -x[1])
        for term, score in sorted_terms[:20]:
            if len(term) >= 4 and term.lower() not in self.STOPWORDS and term.lower() not in skip_terms:
                concepts[term] = {
                    'score': score,
                    'freq': 1,
                    'context': ''
                }
        
        # Method 2: Proper nouns and noun phrases
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for phrase in proper_nouns:
            phrase_lower = phrase.lower()
            # Skip metadata and common terms
            if phrase_lower in self.STOPWORDS or phrase_lower in skip_terms:
                continue
            if len(phrase) <= 3:
                continue
            
            if phrase_lower in concepts:
                concepts[phrase_lower]['score'] += 0.3
                concepts[phrase_lower]['freq'] += 1
            else:
                concepts[phrase_lower] = {
                    'score': 0.5,
                    'freq': 1,
                    'context': ''
                }
        
        # Method 3: "X is a/an Y" patterns for definitions
        definition_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an|the)\s+([^.]+)',
            r'([A-Z][a-z]+)\s+(?:are|was|were)\s+([^.]+)',
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, text)
            for term, definition in matches:
                term_lower = term.lower()
                if term_lower in concepts:
                    concepts[term_lower]['context'] = definition[:150]
                    concepts[term_lower]['score'] += 0.2
                elif len(term) > 3:
                    concepts[term_lower] = {
                        'score': 0.6,
                        'freq': 1,
                        'context': definition[:150]
                    }
        
        # Method 4: Compound terms (X service, X platform, etc.)
        compounds = re.findall(
            r'\b(\w+\s+(?:service|platform|system|network|app|application|feature|tool|method))\b',
            text.lower()
        )
        for compound in compounds:
            if compound not in concepts:
                concepts[compound] = {
                    'score': 0.4,
                    'freq': 1,
                    'context': ''
                }
            else:
                concepts[compound]['score'] += 0.2
        
        # Find context for concepts without definitions
        for term, data in concepts.items():
            if not data['context']:
                for sent in sentences:
                    if term in sent.lower():
                        data['context'] = sent[:200]
                        break
        
        # Sort by score and create ExtractedConcept objects
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
            
            # Create definition
            if data['context']:
                definition = data['context']
            else:
                definition = f"A key concept discussed in this content."
            
            result.append(ExtractedConcept(
                term=term.title(),
                definition=definition,
                frequency=data['freq'],
                importance_score=data['score']
            ))
            seen_terms.add(term)
        
        return result
    
    def _generate_questions(
        self,
        concepts: List[ExtractedConcept],
        sentences: List[str],
        max_questions: int = 8
    ) -> List[GeneratedQuestion]:
        """Generate study questions from concepts"""
        questions = []
        
        # Generate "What is" questions for top concepts
        for i, concept in enumerate(concepts[:4]):
            template = self.QUESTION_TEMPLATES['what_is'][i % len(self.QUESTION_TEMPLATES['what_is'])]
            q = GeneratedQuestion(
                question=template.format(concept=concept.term),
                answer_hint=concept.definition[:100] + "..." if len(concept.definition) > 100 else concept.definition,
                question_type='factual',
                difficulty='easy',
                source_sentence=concept.definition
            )
            questions.append(q)
        
        # Generate "Why" question
        if concepts:
            q = GeneratedQuestion(
                question=f"Why is {concepts[0].term} significant?",
                answer_hint="Consider its role and impact discussed in the content.",
                question_type='conceptual',
                difficulty='medium'
            )
            questions.append(q)
        
        # Generate comparison question if multiple concepts
        if len(concepts) >= 2:
            template = self.QUESTION_TEMPLATES['compare'][0]
            q = GeneratedQuestion(
                question=template.format(concept1=concepts[0].term, concept2=concepts[1].term),
                answer_hint="Look for similarities and differences in their descriptions.",
                question_type='analytical',
                difficulty='medium'
            )
            questions.append(q)
        
        # Generate application question
        if concepts:
            template = self.QUESTION_TEMPLATES['application'][0]
            q = GeneratedQuestion(
                question=template.format(concept=concepts[0].term),
                answer_hint="Think about real-world scenarios where this applies.",
                question_type='application',
                difficulty='hard'
            )
            questions.append(q)
        
        # Generate analysis question
        if concepts:
            template = self.QUESTION_TEMPLATES['analysis'][0]
            q = GeneratedQuestion(
                question=template.format(concept=concepts[0].term),
                answer_hint="Consider both positive and negative aspects.",
                question_type='analytical',
                difficulty='hard'
            )
            questions.append(q)
        
        return questions[:max_questions]
    
    def _find_related_topics(self, text: str, max_topics: int = 5) -> List[str]:
        """Find related topics based on content"""
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score >= 2:
                topic_scores[topic] = score
        
        # Sort by score
        sorted_topics = sorted(topic_scores.items(), key=lambda x: -x[1])
        return [topic for topic, _ in sorted_topics[:max_topics]]
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract action items from text"""
        patterns = [
            r'(\w+)\s+will\s+(\w+.+?)(?:[.!]|$)',
            r'(\w+)\s+should\s+(\w+.+?)(?:[.!]|$)',
            r'(\w+)\s+needs?\s+to\s+(\w+.+?)(?:[.!]|$)',
            r'action\s*(?:item)?[:\s]+(.+?)(?:[.!]|$)',
            r'todo[:\s]+(.+?)(?:[.!]|$)',
        ]
        
        actions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    action = f"{match[0]} will {match[1]}" if len(match) > 1 else match[0]
                else:
                    action = match
                if len(action) > 10 and len(action) < 200:
                    actions.append(action.strip())
        
        return list(set(actions))[:5]
    
    def _extract_decisions(self, text: str) -> List[str]:
        """Extract decisions from text"""
        patterns = [
            r'(?:we\s+)?decided\s+(?:to\s+)?(.+?)(?:[.!]|$)',
            r'decision[:\s]+(.+?)(?:[.!]|$)',
            r'agreed\s+(?:to|on|that)\s+(.+?)(?:[.!]|$)',
            r"let'?s\s+go\s+with\s+(.+?)(?:[.!]|$)",
        ]
        
        decisions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 10 and len(match) < 200:
                    decisions.append(match.strip())
        
        return list(set(decisions))[:5]
    
    def _extract_deadlines(self, text: str) -> List[str]:
        """Extract deadlines from text"""
        patterns = [
            r'by\s+((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))',
            r'by\s+(tomorrow|today|tonight)',
            r'by\s+(next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday))',
            r'by\s+(end\s+of\s+(?:day|week|month))',
            r'deadline[:\s]+(.+?)(?:[.!]|$)',
            r'due\s+(?:by|on)?\s*(.+?)(?:[.!]|$)',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+',
        ]
        
        deadlines = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 2 and len(match) < 100:
                    deadlines.append(match.strip())
        
        return list(set(deadlines))[:5]
