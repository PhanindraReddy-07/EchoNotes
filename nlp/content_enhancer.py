"""
Content Enhancer v3 - Accurate Content Generation
==================================================
Hybrid approach prioritizing EXTRACTION over GENERATION for accuracy.

Key improvements:
1. Extraction-first approach (more accurate)
2. Context-aware AI prompts (when AI is used)
3. Smart sentence scoring and selection
4. Better fallback mechanisms
5. Improved text preprocessing

Modes:
- Extraction-only (default): Fast, accurate, no AI needed
- AI-enhanced: Uses Flan-T5 for additional elaboration
"""
import re
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from nlp.lang_classifier import get_classifier as _get_lang_clf 
from nlp.preprocessor import TextPreprocessor
_preprocessor = TextPreprocessor(remove_fillers=True)

@dataclass
class GeneratedContent:
    """Container for all generated/extracted content"""
    original: str
    simplified_explanation: str
    key_takeaways: List[str]
    elaboration: str
    examples: List[str]
    faq: List[Dict[str, str]]
    vocabulary: List[Dict[str, str]]
    
    def to_dict(self) -> Dict:
        return {
            'simplified': self.simplified_explanation,
            'takeaways': self.key_takeaways,
            'elaboration': self.elaboration,
            'examples': self.examples,
            'faq': self.faq,
            'vocabulary': self.vocabulary
        }


class ContentEnhancer:
    """
    Content enhancement using extraction-first approach.
    
    This class extracts and structures content from the source text,
    optionally using AI models for additional elaboration.
    
    The extraction-first approach is more accurate because it uses
    the actual content rather than generating potentially irrelevant text.
    """
    
    # Stopwords for filtering
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
        'i', 'me', 'my', 'as', 'if', 'so', 'than', 'such', 'when', 'where',
        'which', 'who', 'what', 'how', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'any', 'no', 'not', 'only', 'same',
        'just', 'also', 'very', 'even', 'back', 'now', 'well', 'also', 'just',
        'like', 'really', 'want', 'going', 'something', 'actually', 'thing',
        'things', 'way', 'yeah', 'yes', 'okay', 'ok', 'um', 'uh', 'basically',
    }
    
    # Important sentence indicators (for scoring)
    IMPORTANCE_SIGNALS = {
        'high': [
            'in conclusion', 'to summarize', 'the main point', 'most importantly',
            'the key is', 'crucial', 'essential', 'fundamental', 'primarily',
            'significantly', 'notably', 'the purpose', 'the goal', 'therefore',
            'as a result', 'in summary', 'ultimately', 'in essence', 'the bottom line',
            'this means', 'this shows', 'this demonstrates', 'importantly',
        ],
        'medium': [
            'for example', 'for instance', 'such as', 'including', 'specifically',
            'first', 'second', 'third', 'finally', 'additionally', 'moreover',
            'furthermore', 'however', 'because', 'since', 'due to', 'leads to',
            'according to', 'research shows', 'studies indicate', 'in other words',
        ],
        'definition': [
            'is defined as', 'refers to', 'means that', 'is called', 'known as',
            'is a', 'is an', 'are a', 'are an', 'can be described as',
        ],
        'example': [
            'for example', 'for instance', 'such as', 'like', 'including',
            'consider', 'imagine', 'suppose', 'take the case of', 'as an example',
        ],
    }
    
    def __init__(self, use_ai: bool = False, model_name: str = "google/flan-t5-base"):
        """
        Initialize content enhancer
        
        Args:
            use_ai: Whether to use AI model for additional generation
            model_name: HuggingFace model to use if AI enabled
        """
        self.use_ai = use_ai
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.cache_dir = Path.home() / ".cache" / "echonotes"
        self._model_loaded = False
    
    def _load_model(self) -> bool:
        """Load AI model if needed and not already loaded"""
        if not self.use_ai:
            return False
        
        if self._model_loaded:
            return True
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            print(f"[ContentEnhancer] Loading model: {self.model_name}")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(device)
            
            self.device = device
            self._model_loaded = True
            print(f"[ContentEnhancer] Model loaded on {device}")
            return True
            
        except ImportError:
            print("[ContentEnhancer] AI disabled - transformers/torch not installed")
            self.use_ai = False
            return False
        except Exception as e:
            print(f"[ContentEnhancer] Error loading model: {e}")
            self.use_ai = False
            return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        lines = text.split('\n')
        clean_lines = []
        
        # Skip patterns for metadata
        skip_patterns = [
            r'^={2,}', r'^-{2,}', r'^\[.*\]$',
            r'^transcript:', r'^audio:', r'^duration:',
            r'^confidence:', r'^words:', r'^timestamps:',
            r'^generated:', r'^echonotes', r'^recording',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip metadata
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
            
            # Remove timestamp markers
            line = re.sub(r'\[\d{1,2}:\d{2}(?::\d{2})?\]', '', line).strip()
            
            if len(line) > 5:
                clean_lines.append(line)
        
        text = ' '.join(clean_lines)
        
        # Normalize whitespace
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
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in self.STOPWORDS]
    
    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        """Score sentence by importance using multiple signals"""
        score = 0.0
        sent_lower = sentence.lower()
        
        # Position score (first/last sentences are important)
        rel_pos = position / max(1, total - 1)
        if rel_pos < 0.15:
            score += 0.3
        elif rel_pos > 0.85:
            score += 0.2
        
        # Length score (prefer medium length)
        words = sentence.split()
        if 12 <= len(words) <= 35:
            score += 0.2
        elif len(words) < 8 or len(words) > 50:
            score -= 0.1
        
        # Importance signal score
        for phrase in self.IMPORTANCE_SIGNALS['high']:
            if phrase in sent_lower:
                score += 0.4
                break
        
        for phrase in self.IMPORTANCE_SIGNALS['medium']:
            if phrase in sent_lower:
                score += 0.2
                break
        
        # Definition pattern score
        for phrase in self.IMPORTANCE_SIGNALS['definition']:
            if phrase in sent_lower:
                score += 0.3
                break
        
        return score
    
    def _extract_key_sentences(self, sentences: List[str], max_count: int = 5) -> List[str]:
        """Extract most important sentences"""
        if len(sentences) <= max_count:
            return sentences
        
        # Score all sentences
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences))
            scored.append((i, sent, score))
        
        # Sort by score and select top
        scored.sort(key=lambda x: -x[2])
        selected = scored[:max_count]
        
        # Sort by position for coherent reading
        selected.sort(key=lambda x: x[0])
        
        return [sent for _, sent, _ in selected]
    
    def _extract_examples(self, sentences: List[str], max_count: int = 3) -> List[str]:
        """Extract example sentences from text"""
        examples = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            for phrase in self.IMPORTANCE_SIGNALS['example']:
                if phrase in sent_lower:
                    # Clean and add
                    example = sent.strip()
                    if example not in examples and len(example) > 20:
                        examples.append(example)
                        break
            
            if len(examples) >= max_count:
                break
        
        return examples
    
    def _extract_definitions(self, sentences: List[str]) -> List[Dict[str, str]]:
        """Extract term definitions from text"""
        definitions = []
        
        # Patterns for definitions
        patterns = [
            r'(\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:is|are)\s+(?:a|an|the)?\s*(.+?)(?:\.|,)',
            r'(\b[A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+refers to\s+(.+?)(?:\.|,)',
            r'(\b[A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+means\s+(.+?)(?:\.|,)',
            r'(\b[A-Z][a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+can be defined as\s+(.+?)(?:\.|,)',
        ]
        
        for sent in sentences:
            for pattern in patterns:
                matches = re.findall(pattern, sent)
                for term, definition in matches:
                    term = term.strip()
                    definition = definition.strip()
                    
                    # Validate
                    if (len(term) > 2 and len(definition) > 10 and 
                        len(definition) < 200 and
                        term.lower() not in self.STOPWORDS):
                        
                        # Check not already added
                        if not any(d['term'].lower() == term.lower() for d in definitions):
                            definitions.append({
                                'term': term,
                                'meaning': definition[0].upper() + definition[1:] + '.' if not definition.endswith('.') else definition[0].upper() + definition[1:]
                            })
        
        return definitions[:10]
    
    def _extract_vocabulary(self, text: str, sentences: List[str], max_terms: int = 8) -> List[Dict[str, str]]:
        """Extract key vocabulary terms with context"""
        # First try to find explicit definitions
        definitions = self._extract_definitions(sentences)
        
        # Find additional key terms
        text_lower = text.lower()
        
        # Extract capitalized terms (likely important)
        terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        term_freq = Counter(terms)
        
        # Also extract technical terms
        technical_patterns = [
            r'\b([a-z]+(?:tion|ment|ity|ism|ology|graphy))\b',
            r'\b([a-z]+\s+(?:system|method|process|model|theory|approach))\b',
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text_lower)
            for term in matches:
                term_freq[term.title()] += 1
        
        # Build vocabulary list
        vocabulary = list(definitions)  # Start with found definitions
        seen_terms = {d['term'].lower() for d in definitions}
        
        # Add high-frequency terms without definitions
        for term, freq in term_freq.most_common(20):
            if term.lower() in seen_terms or term.lower() in self.STOPWORDS:
                continue
            if len(term) < 3 or freq < 2:
                continue
            
            # Find context sentence for this term
            context = ""
            for sent in sentences:
                if term.lower() in sent.lower():
                    context = sent
                    break
            
            if context:
                # Truncate to max 15 words for concise definition
                cwords = context.split()
                if len(cwords) > 15:
                    context = ' '.join(cwords[:15]) + '...'
                
                vocabulary.append({
                    'term': term,
                    'meaning': context
                })
                seen_terms.add(term.lower())
        
        return vocabulary[:max_terms]
    
    def _generate_faq(self, text: str, sentences: List[str], concepts: List[str]) -> List[Dict[str, str]]:
        """Generate FAQ based on content extraction - multilingual support"""
        faq = []
        
        def _short(sent, max_w=25):
            """Truncate a sentence to max words"""
            w = sent.split()[:max_w]
            s = ' '.join(w)
            return s + ('...' if len(w) == max_w else '')
        
        # Detect language for keyword matching
        _lang = _get_lang_clf().predict(text[:300])
        is_hindi  = _lang.label in ('hi', 'hi_en')
        is_telugu = _lang.label in ('te', 'te_en')
        
        # Q1: Main topic
        if sentences:
            main_topic = concepts[0] if concepts else ("इस विषय" if is_hindi else "ఈ అంశం" if is_telugu else "this topic")
            faq.append({
                'q': f"{'यह क्या है' if is_hindi else 'ఇది ఏమిటి' if is_telugu else 'What is'}: {main_topic}?",
                'a': _short(sentences[0])
            })
        
        # Q2: Why important
        why_kw = (['क्योंकि', 'कारण', 'महत्वपूर्ण', 'आवश्यक', 'जरूरी'] if is_hindi
                   else ['ఎందుకంటే', 'కారణం', 'ముఖ్యమైన', 'అవసరం', 'ప్రాముఖ్యత'] if is_telugu
                   else ['because', 'reason', 'purpose', 'important', 'significant', 'crucial'])
        for sent in sentences[1:]:
            if any(w in sent.lower() if not (is_hindi or is_telugu) else w in sent for w in why_kw):
                q = "यह क्यों महत्वपूर्ण है?" if is_hindi else "ఇది ఎందుకు ముఖ్యమైనది?" if is_telugu else "Why is this important?"
                faq.append({'q': q, 'a': _short(sent)})
                break
        
        # Q3: How it works
        how_kw = (['द्वारा', 'तरीका', 'प्रक्रिया', 'उपयोग', 'माध्यम'] if is_hindi
                   else ['ద్వారా', 'పద్ధతి', 'ప్రక్రియ', 'ఉపయోగించి'] if is_telugu
                   else ['by', 'through', 'using', 'process', 'method', 'step'])
        for sent in sentences:
            if sent == sentences[0]:
                continue
            if any(w in sent.lower() if not (is_hindi or is_telugu) else w in sent for w in how_kw):
                q = "यह कैसे काम करता है?" if is_hindi else "ఇది ఎలా పని చేస్తుంది?" if is_telugu else "How does this work?"
                faq.append({'q': q, 'a': _short(sent)})
                break
        
        # Q4: Benefits/advantages
        ben_kw = (['लाभ', 'फायदा', 'सुविधा', 'बेहतर', 'सक्षम'] if is_hindi
                   else ['ప్రయోజనం', 'లాభం', 'సౌలభ్యం', 'మెరుగు'] if is_telugu
                   else ['benefit', 'advantage', 'allows', 'enables', 'provides', 'improve'])
        for sent in sentences:
            if any(w in sent.lower() if not (is_hindi or is_telugu) else w in sent for w in ben_kw):
                if sent not in [f['a'].rstrip('...') for f in faq]:
                    q = "मुख्य लाभ क्या हैं?" if is_hindi else "ప్రధాన ప్రయోజనాలు ఏమిటి?" if is_telugu else "What are the key benefits?"
                    faq.append({'q': q, 'a': _short(sent)})
                    break
        
        # Q5: Challenges/problems
        chal_kw = (['चुनौती', 'समस्या', 'कठिनाई', 'कमी', 'बाधा'] if is_hindi
                    else ['సవాలు', 'సమస్య', 'కష్టం', 'లేమి', 'అడ్డంకి'] if is_telugu
                    else ['challenge', 'problem', 'difficulty', 'risk', 'limitation', 'issue'])
        for sent in sentences:
            if any(w in sent.lower() if not (is_hindi or is_telugu) else w in sent for w in chal_kw):
                if sent not in [f['a'].rstrip('...') for f in faq]:
                    q = "मुख्य चुनौतियाँ क्या हैं?" if is_hindi else "ప్రధాన సవాళ్ళు ఏమిటి?" if is_telugu else "What are the main challenges?"
                    faq.append({'q': q, 'a': _short(sent)})
                    break
        
        # If still few FAQs, add concept-based questions
        for concept in concepts[1:4]:
            if len(faq) >= 5:
                break
            # Find sentence mentioning this concept
            for sent in sentences:
                if concept.lower() in sent.lower() if not (is_hindi or is_telugu) else concept in sent:
                    q_word = "क्या है" if is_hindi else "ఏమిటి" if is_telugu else "What is"
                    faq.append({'q': f"{q_word} {concept}?", 'a': _short(sent)})
                    break
        
        return faq[:5]
    
    def _ai_generate(self, prompt: str, max_length: int = 150) -> str:
        """Generate text using AI model (if available)"""
        if not self._load_model():
            return ""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.3,
                do_sample=True,
                early_stopping=True
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
            
        except Exception as e:
            print(f"[ContentEnhancer] AI generation error: {e}")
            return ""
    
    def simplify(self, text: str) -> str:
        """Create a SHORT simplified explanation (2-3 sentences max, ~50-80 words)"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        if not sentences:
            return ""
        
        # Extract only the 2 most important sentences for a brief summary
        key_sents = self._extract_key_sentences(sentences, max_count=2)
        
        if not key_sents:
            key_sents = sentences[:2]
        
        # Truncate each sentence to max 40 words for brevity
        brief = []
        total_words = 0
        for sent in key_sents:
            words = sent.split()
            if total_words + len(words) > 80:
                remaining = 80 - total_words
                if remaining > 8:
                    brief.append(' '.join(words[:remaining]) + '...')
                break
            brief.append(sent.strip())
            total_words += len(words)
        
        return ' '.join(brief) if brief else sentences[0][:300]
    
    def generate_takeaways(self, text: str, num_points: int = 5) -> List[str]:
        """Extract SHORT key takeaways (each max 25 words, distinct from summary)"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        if not sentences:
            return []
        
        # Score ALL sentences and pick top ones
        scored = []
        for i, sent in enumerate(sentences):
            score = self._score_sentence(sent, i, len(sentences))
            scored.append((score, sent))
        scored.sort(key=lambda x: -x[0])
        
        takeaways = []
        seen_words = set()
        for _, sent in scored:
            # Truncate to max 25 words per takeaway
            words = sent.split()[:25]
            takeaway = ' '.join(words)
            if len(words) == 25 and not takeaway.endswith(('.', '!', '?', '।')):
                takeaway += '...'
            
            # Skip if too similar to existing takeaways (deduplication)
            tw = set(w.lower() for w in words if len(w) > 3)
            if seen_words and len(tw & seen_words) / max(len(tw), 1) > 0.5:
                continue
            seen_words.update(tw)
            
            if takeaway and len(takeaway) > 15:
                if takeaway[0].isalpha() and not takeaway[0].isupper():
                    takeaway = takeaway[0].upper() + takeaway[1:]
                takeaways.append(takeaway)
            
            if len(takeaways) >= num_points:
                break
        
        return takeaways
    
    def elaborate(self, text: str) -> str:
        """Create elaboration of the content"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        if not sentences:
            return ""
        
        # Find sentences with definitions or explanations
        elaboration_sents = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(phrase in sent_lower for phrase in self.IMPORTANCE_SIGNALS['definition']):
                elaboration_sents.append(sent)
            elif any(phrase in sent_lower for phrase in ['means', 'explains', 'describes', 'shows']):
                elaboration_sents.append(sent)
        
        if elaboration_sents:
            return ' '.join(elaboration_sents[:3])
        
        # Fallback: return first few sentences
        return ' '.join(sentences[:2])
    
    def generate_examples(self, text: str, num_examples: int = 3) -> List[str]:
        """Extract example-like sentences from text (max 30 words each)"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        examples = self._extract_examples(sentences, num_examples)
        
        # If not enough explicit examples, find sentences with specific details
        if len(examples) < num_examples:
            for sent in sentences:
                if sent not in examples and len(sent.split()) >= 8:
                    # Prefer sentences with numbers, names, or specifics
                    has_specifics = bool(re.search(r'\d+|\b[A-Z][a-z]+\b', sent))
                    if has_specifics or len(examples) < 1:
                        # Truncate to 30 words
                        words = sent.split()[:30]
                        ex = ' '.join(words)
                        if len(words) == 30:
                            ex += '...'
                        examples.append(ex)
                if len(examples) >= num_examples:
                    break
        
        return examples[:num_examples]
    
    def generate_faq(self, text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """Generate FAQ from text"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        # Extract main concepts - try English capitalized words first
        concepts = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        concepts = list(dict.fromkeys(concepts))[:5]
        
        # Fallback for Hindi/Telugu: use frequent content words
        if not concepts:
            from collections import Counter
            stopwords = {
                # Hindi
                'है', 'हैं', 'का', 'के', 'की', 'में', 'से', 'को', 'और', 'पर',
                'ने', 'या', 'एक', 'यह', 'इस', 'जो', 'तो', 'भी', 'था', 'कि',
                # Telugu
                'మరియు', 'ఈ', 'ఆ', 'ఇది', 'అది', 'లో', 'కు', 'తో', 'గా', 'ఉంది',
                # English — expanded significantly
                'the', 'is', 'are', 'and', 'of', 'to', 'in', 'a', 'an', 'for',
                'it', 'its', 'this', 'that', 'these', 'those', 'all', 'my', 'we',
                'you', 'your', 'our', 'they', 'them', 'their', 'he', 'she', 'his',
                'her', 'me', 'us', 'who', 'what', 'which', 'how', 'when', 'where',
                'why', 'be', 'been', 'was', 'were', 'have', 'has', 'had', 'do',
                'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'just', 'also', 'very', 'not', 'no', 'so', 'but', 'or', 'if',
                'as', 'at', 'by', 'on', 'up', 'out', 'get', 'got', 'can', 'one',
                'with', 'from', 'about', 'into', 'like', 'than', 'then', 'more',
                'some', 'any', 'each', 'only', 'such', 'same', 'well', 'back',
                'bring', 'brings', 'brought', 'make', 'makes', 'made', 'take',
                'takes', 'took', 'come', 'comes', 'came', 'see', 'sees', 'saw',
                'know', 'knew', 'think', 'thought', 'want', 'wanted', 'need',
                'say', 'said', 'tell', 'told', 'use', 'used', 'play', 'plays',
                'draw', 'draws', 'watch', 'hear', 'heard', 'trust', 'deliver',
                'unlike', 'instead', 'however', 'although', 'whereas',
                'therefore', 'furthermore', 'moreover', 'similarly',
            }
            word_freq = Counter()
            for w in clean_text.split():
                cw = re.sub(r'[।,.!?:;()\[\]{}"\'/\\-]', '', w).strip()
                if cw and len(cw) >= 3 and cw.lower() not in stopwords and cw not in stopwords:
                    word_freq[cw] += 1
            concepts = [w for w, _ in word_freq.most_common(15) if _ >= 2 and len(w) >= 5][:5]
        
        return self._generate_faq(clean_text, sentences, concepts)
    
    def extract_vocabulary(self, text: str, num_terms: int = 8) -> List[Dict[str, str]]:
        """Extract vocabulary terms with definitions"""
        clean_text = self._clean_text(text)
        sentences = self._split_sentences(clean_text)
        
        return self._extract_vocabulary(clean_text, sentences, num_terms)
    
    def enhance_content(self, text: str, title: str = "Content") -> GeneratedContent:
        """
        Generate all enhanced content for a text
        
        Uses extraction-first approach for accuracy,
        with optional AI enhancement.
        """
        print(f"\n🔄 Generating enhanced content...")
        
        text = _preprocessor.clean_for_nlp(text)
        clean_text = self._clean_text(text)
        
        if not clean_text or len(clean_text) < 20:
            print("   ⚠️ Not enough content to enhance")
            return GeneratedContent(
                original=text,
                simplified_explanation="",
                key_takeaways=[],
                elaboration="",
                examples=[],
                faq=[],
                vocabulary=[]
            )
        
        print(f"   📄 Processing {len(clean_text)} characters...")
        
        print("   📝 Extracting simplified explanation...")
        simplified = self.simplify(clean_text)
        
        print("   🎯 Extracting key takeaways...")
        takeaways = self.generate_takeaways(clean_text, 5)
        
        print("   📖 Creating elaboration...")
        elaboration = self.elaborate(clean_text)
        
        print("   💡 Finding examples...")
        examples = self.generate_examples(clean_text, 3)
        
        print("   ❓ Generating FAQ...")
        faq = self.generate_faq(clean_text, 5)
        
        print("   📚 Extracting vocabulary...")
        vocabulary = self.extract_vocabulary(clean_text, 8)
        
        print("   ✅ Content enhancement complete!")
        
        return GeneratedContent(
            original=text,
            simplified_explanation=simplified,
            key_takeaways=takeaways,
            elaboration=elaboration,
            examples=examples,
            faq=faq,
            vocabulary=vocabulary
        )


class OfflineContentGenerator:
    """Alias for backward compatibility"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self._enhancer = ContentEnhancer(use_ai=True, model_name=model_name)
    
    def simplify(self, text: str, target_level: str = "high school") -> str:
        return self._enhancer.simplify(text)
    
    def elaborate(self, text: str) -> str:
        return self._enhancer.elaborate(text)
    
    def explain_like_im_5(self, text: str) -> str:
        return self._enhancer.simplify(text)
    
    def generate_takeaways(self, text: str, num_points: int = 5) -> List[str]:
        return self._enhancer.generate_takeaways(text, num_points)
    
    def generate_examples(self, concept: str, num_examples: int = 3) -> List[str]:
        return self._enhancer.generate_examples(concept, num_examples)
    
    def generate_faq(self, text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        return self._enhancer.generate_faq(text, num_questions)
    
    def extract_vocabulary(self, text: str, num_terms: int = 10) -> List[Dict[str, str]]:
        return self._enhancer.extract_vocabulary(text, num_terms)
    
    def enhance_content(self, text: str, title: str = "Content") -> GeneratedContent:
        return self._enhancer.enhance_content(text, title)


def get_content_enhancer(use_ai: bool = False) -> ContentEnhancer:
    """
    Factory function to get content enhancer
    
    Args:
        use_ai: Whether to enable AI-powered generation
        
    Returns:
        ContentEnhancer instance
    """
    return ContentEnhancer(use_ai=use_ai)
