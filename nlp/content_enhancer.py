"""
Generative Content Enhancer - Offline NLP
==========================================
Uses offline transformer models to generate:
- Simple explanations for complex concepts
- Elaborated content for better understanding
- Key takeaways and insights
- Simplified summaries for different audiences
- Related examples and analogies
- FAQ generation

Models used (all offline after download):
- google/flan-t5-small (77MB) - Best for instructions/Q&A
- t5-small (242MB) - Good for summarization/generation
- sentence-transformers/all-MiniLM-L6-v2 (23MB) - Semantic similarity

Usage:
    enhancer = ContentEnhancer()
    
    # Generate simple explanation
    simple = enhancer.simplify("Complex technical text here...")
    
    # Generate elaboration
    detailed = enhancer.elaborate("Brief concept description")
    
    # Generate key takeaways
    takeaways = enhancer.generate_takeaways(text)
"""

import re
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GeneratedContent:
    """Container for generated content"""
    original: str
    simplified_explanation: str
    key_takeaways: List[str]
    elaboration: str
    eli5_explanation: str  # Explain Like I'm 5
    examples: List[str]
    faq: List[Dict[str, str]]  # [{'q': question, 'a': answer}]
    vocabulary: List[Dict[str, str]]  # [{'term': word, 'meaning': definition}]
    
    def to_dict(self) -> Dict:
        return {
            'simplified': self.simplified_explanation,
            'takeaways': self.key_takeaways,
            'elaboration': self.elaboration,
            'eli5': self.eli5_explanation,
            'examples': self.examples,
            'faq': self.faq,
            'vocabulary': self.vocabulary
        }


class ContentEnhancer:
    """
    Generates enhanced educational content using offline models
    
    Features:
    - Simplify complex text
    - Generate key takeaways
    - Create ELI5 (Explain Like I'm 5) versions
    - Generate relevant examples
    - Create FAQ from content
    - Extract and define vocabulary
    
    All models run 100% offline after initial download.
    """
    
    # Model configurations
    MODELS = {
        'flan-t5': {
            'name': 'google/flan-t5-base',  # Upgraded from small (250MB vs 77MB)
            'size': '250MB',
            'task': 'text2text-generation'
        },
        'flan-t5-small': {
            'name': 'google/flan-t5-small',
            'size': '77MB',
            'task': 'text2text-generation'
        },
        't5': {
            'name': 't5-small', 
            'size': '242MB',
            'task': 'text2text-generation'
        },
        'embeddings': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'size': '23MB'
        }
    }
    
    def __init__(
        self,
        model_name: str = 'flan-t5',
        cache_dir: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Initialize content enhancer
        
        Args:
            model_name: 'flan-t5' (recommended) or 't5'
            cache_dir: Directory to cache models
            device: 'auto', 'cpu', or 'cuda'
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or str(Path.home() / '.cache' / 'echonotes')
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.embedder = None
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_model(self) -> bool:
        """Load the text generation model"""
        if self.model is not None:
            return True
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            model_config = self.MODELS.get(self.model_name, self.MODELS['flan-t5'])
            model_name = model_config['name']
            
            print(f"[ContentEnhancer] Loading {model_name}...")
            print(f"[ContentEnhancer] This may take a moment on first run (downloading {model_config['size']})")
            
            # Determine device
            if self.device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            ).to(device)
            
            self.device = device
            print(f"[ContentEnhancer] Model loaded on {device}")
            return True
            
        except ImportError:
            print("[ContentEnhancer] Required: pip install transformers torch")
            return False
        except Exception as e:
            print(f"[ContentEnhancer] Error loading model: {e}")
            return False
    
    def _clean_input_text(self, text: str) -> str:
        """Clean transcript metadata from input text"""
        lines = text.split('\n')
        clean_lines = []
        
        skip_keywords = [
            'echonotes', 'transcript', 'audio:', 'duration:', 'words:', 
            'confidence:', 'timestamps:', 'generated:', 'recording'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip metadata lines
            line_lower = line.lower()
            if any(kw in line_lower for kw in skip_keywords):
                continue
            if line.startswith('=') or line.startswith('-'):
                continue
            if re.match(r'^\[\d{2}:\d{2}\]', line):
                continue
            
            # Remove timestamp markers
            line = re.sub(r'\[\d{2}:\d{2}\]', '', line).strip()
            
            if len(line) > 10:
                clean_lines.append(line)
        
        return ' '.join(clean_lines)
    
    def _generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        num_beams: int = 4
    ) -> str:
        """Generate text from prompt"""
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
                num_beams=num_beams,
                temperature=temperature,
                do_sample=temperature > 0,
                early_stopping=True
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
            
        except Exception as e:
            print(f"[ContentEnhancer] Generation error: {e}")
            return ""
    
    def simplify(self, text: str, target_level: str = "high school") -> str:
        """
        Simplify complex text for easier understanding
        
        Args:
            text: Complex text to simplify
            target_level: "elementary", "middle school", "high school", "general"
            
        Returns:
            Simplified explanation
        """
        # Better prompt for Flan-T5
        prompt = f"Summarize and explain this simply: {text[:500]}"
        result = self._generate(prompt, max_length=200)
        return result if result else "This content explains the key topic and its main features."
    
    def elaborate(self, text: str) -> str:
        """
        Elaborate on brief content with more details
        
        Args:
            text: Brief text to elaborate
            
        Returns:
            More detailed explanation
        """
        prompt = f"Provide more details about: {text[:300]}"
        result = self._generate(prompt, max_length=300)
        return result if result else text
    
    def explain_like_im_5(self, text: str) -> str:
        """
        Generate ELI5 (Explain Like I'm 5) explanation
        
        Args:
            text: Text to explain simply
            
        Returns:
            Very simple explanation
        """
        # Simple prompt that works well with Flan-T5
        prompt = f"Explain this to a child: {text[:300]}"
        result = self._generate(prompt, max_length=150)
        return result if result else "This is about something interesting that helps people do things easier."
    
    def generate_takeaways(self, text: str, num_points: int = 5) -> List[str]:
        """
        Generate key takeaways from content
        
        Args:
            text: Source text
            num_points: Number of takeaways to generate
            
        Returns:
            List of key takeaway points
        """
        prompt = f"What are the main points? {text[:500]}"
        result = self._generate(prompt, max_length=300)
        
        if not result:
            # Fallback: extract from original text
            sentences = text.split('.')
            return [s.strip() for s in sentences[:num_points] if len(s.strip()) > 10]
        
        # Parse into list
        takeaways = []
        lines = result.replace('. ', '.\n').split('\n')
        for line in lines:
            line = line.strip()
            # Remove numbering
            line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
            if line and len(line) > 10:
                takeaways.append(line)
        
        # If no proper list, split by sentences
        if not takeaways:
            sentences = re.split(r'[.!?]+', result)
            takeaways = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return takeaways[:num_points] if takeaways else ["Key information is presented in the content."]
    
    def generate_examples(self, concept: str, num_examples: int = 3) -> List[str]:
        """
        Generate real-world examples for a concept
        
        Args:
            concept: The concept to exemplify
            num_examples: Number of examples to generate
            
        Returns:
            List of examples
        """
        prompt = f"Give examples of: {concept[:200]}"
        result = self._generate(prompt, max_length=250)
        
        if not result:
            return [f"Example: {concept} is used in everyday applications."]
        
        # Parse examples
        examples = []
        lines = result.replace('. ', '.\n').split('\n')
        for line in lines:
            line = line.strip()
            line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
            if line and len(line) > 10:
                examples.append(line)
        
        return examples[:num_examples] if examples else [result]
    
    def generate_faq(self, text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """
        Generate FAQ from content
        
        Args:
            text: Source text
            num_questions: Number of Q&A pairs
            
        Returns:
            List of {'q': question, 'a': answer} dicts
        """
        text_short = text[:500]
        
        # Generate Q&A pairs one by one for better quality
        faq = []
        
        # Q1: What is it?
        q1 = "What is this about?"
        a1 = self._generate(f"Answer: What is this about? {text_short}", max_length=100)
        if a1:
            faq.append({'q': q1, 'a': a1})
        
        # Q2: How does it work?
        q2 = "How does it work?"
        a2 = self._generate(f"Answer: How does this work? {text_short}", max_length=100)
        if a2:
            faq.append({'q': q2, 'a': a2})
        
        # Q3: Why is it important?
        q3 = "Why is this important?"
        a3 = self._generate(f"Answer: Why is this important? {text_short}", max_length=100)
        if a3:
            faq.append({'q': q3, 'a': a3})
        
        # Q4: What are the key features?
        q4 = "What are the key features?"
        a4 = self._generate(f"What are the features? {text_short}", max_length=100)
        if a4:
            faq.append({'q': q4, 'a': a4})
        
        return faq[:num_questions]
    
    def extract_vocabulary(self, text: str, num_terms: int = 10) -> List[Dict[str, str]]:
        """
        Extract and define key vocabulary terms
        
        Args:
            text: Source text
            num_terms: Number of terms to extract
            
        Returns:
            List of {'term': word, 'meaning': definition}
        """
        # Find potential vocabulary (capitalized terms, technical words)
        words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        technical = re.findall(r'\b([a-z]+(?:tion|ment|ity|ism|ology|graphy|ing))\b', text.lower())
        
        # Combine and deduplicate
        candidates = list(dict.fromkeys(words + [t.title() for t in technical]))[:num_terms * 2]
        
        vocabulary = []
        for term in candidates[:num_terms]:
            if len(term) < 3:
                continue
            prompt = f"Define {term}:"
            definition = self._generate(prompt, max_length=80)
            if definition and len(definition) > 5:
                vocabulary.append({
                    'term': term,
                    'meaning': definition
                })
            else:
                # Fallback definition
                vocabulary.append({
                    'term': term,
                    'meaning': f"A key term related to the main topic."
                })
        
        return vocabulary[:num_terms]
    
    def generate_analogy(self, concept: str) -> str:
        """
        Generate an analogy to explain a concept
        
        Args:
            concept: Concept to explain
            
        Returns:
            Analogy explanation
        """
        prompt = f"Explain {concept} using a simple everyday analogy:"
        return self._generate(prompt, max_length=150)
    
    def summarize_for_audience(
        self,
        text: str,
        audience: str = "general"
    ) -> str:
        """
        Summarize text for specific audience
        
        Args:
            text: Text to summarize
            audience: "expert", "professional", "student", "general", "child"
            
        Returns:
            Audience-appropriate summary
        """
        prompts = {
            'expert': f"Summarize this for an expert audience with technical details: {text}",
            'professional': f"Summarize this for business professionals: {text}",
            'student': f"Summarize this for a student studying the topic: {text}",
            'general': f"Summarize this for a general audience in simple terms: {text}",
            'child': f"Summarize this for a child in very simple words: {text}"
        }
        
        prompt = prompts.get(audience, prompts['general'])
        return self._generate(prompt, max_length=200)
    
    def enhance_content(self, text: str, title: str = "Content") -> GeneratedContent:
        """
        Generate all enhanced content for a text
        
        Args:
            text: Source text
            title: Content title
            
        Returns:
            GeneratedContent with all generated sections
        """
        print(f"\n🔄 Generating enhanced content...")
        
        # Clean text first - remove transcript metadata
        clean_text = self._clean_input_text(text)
        
        # Truncate if too long
        text_short = clean_text[:1000] if len(clean_text) > 1000 else clean_text
        
        if not text_short or len(text_short) < 20:
            print("   ⚠️ Not enough content to enhance")
            return GeneratedContent(
                original=text,
                simplified_explanation="",
                key_takeaways=[],
                elaboration="",
                eli5_explanation="",
                examples=[],
                faq=[],
                vocabulary=[]
            )
        
        print(f"   📄 Processing {len(text_short)} characters...")
        
        print("   📝 Generating simplified explanation...")
        simplified = self.simplify(text_short)
        
        print("   🎯 Generating key takeaways...")
        takeaways = self.generate_takeaways(text_short, 5)
        
        print("   📖 Generating elaboration...")
        elaboration = self.elaborate(text_short[:500])
        
        print("   👶 Generating ELI5 explanation...")
        eli5 = self.explain_like_im_5(text_short[:300])
        
        print("   💡 Generating examples...")
        # Extract main concept for examples
        first_sentence = text_short.split('.')[0] if '.' in text_short else text_short[:100]
        examples = self.generate_examples(first_sentence, 3)
        
        print("   ❓ Generating FAQ...")
        faq = self.generate_faq(text_short, 4)
        
        print("   📚 Extracting vocabulary...")
        vocabulary = self.extract_vocabulary(clean_text, 8)
        
        print("   ✅ Content generation complete!")
        
        return GeneratedContent(
            original=text,
            simplified_explanation=simplified,
            key_takeaways=takeaways,
            elaboration=elaboration,
            eli5_explanation=eli5,
            examples=examples,
            faq=faq,
            vocabulary=vocabulary
        )


class OfflineContentGenerator:
    """
    Fallback generator when transformers not available
    Uses rule-based methods for content generation
    """
    
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'this', 'that', 'it', 'they', 'we', 'you', 'he', 'she'
        }
    
    def simplify(self, text: str) -> str:
        """Simple rule-based simplification"""
        sentences = re.split(r'[.!?]+', text)
        simplified = []
        
        for sent in sentences[:3]:
            sent = sent.strip()
            if len(sent) > 10:
                # Keep shorter sentences
                words = sent.split()
                if len(words) > 20:
                    sent = ' '.join(words[:20]) + '...'
                simplified.append(sent)
        
        return '. '.join(simplified) + '.' if simplified else text[:200]
    
    def generate_takeaways(self, text: str, num: int = 5) -> List[str]:
        """Extract key sentences as takeaways"""
        sentences = re.split(r'[.!?]+', text)
        
        # Score sentences
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:
                # Score by position and keywords
                score = 0
                if 'important' in sent.lower() or 'key' in sent.lower():
                    score += 2
                if sentences.index(sent + '.') == 0:
                    score += 1
                scored.append((score, sent))
        
        scored.sort(reverse=True)
        return [s for _, s in scored[:num]]
    
    def generate_questions(self, text: str, num: int = 5) -> List[Dict[str, str]]:
        """Generate simple questions from content"""
        sentences = re.split(r'[.!?]+', text)
        questions = []
        
        templates = [
            ("What is", "?"),
            ("How does", " work?"),
            ("Why is", " important?"),
            ("What are the features of", "?"),
            ("Explain", "."),
        ]
        
        # Extract nouns/concepts
        concepts = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        
        for i, concept in enumerate(concepts[:num]):
            template = templates[i % len(templates)]
            q = f"{template[0]} {concept}{template[1]}"
            
            # Find answer in text
            answer = ""
            for sent in sentences:
                if concept.lower() in sent.lower():
                    answer = sent.strip()
                    break
            
            questions.append({'q': q, 'a': answer or f"See the content about {concept}."})
        
        return questions
    
    def extract_vocabulary(self, text: str, num: int = 8) -> List[Dict[str, str]]:
        """Extract vocabulary with simple definitions"""
        # Find capitalized terms and technical words
        terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        unique_terms = list(dict.fromkeys(terms))  # Remove duplicates, keep order
        
        vocab = []
        sentences = text.split('.')
        
        for term in unique_terms[:num]:
            # Find definition in context
            definition = f"A concept related to {term.lower()}"
            for sent in sentences:
                if term in sent:
                    # Use the sentence as context
                    definition = sent.strip()[:100]
                    break
            
            vocab.append({'term': term, 'meaning': definition})
        
        return vocab
    
    def enhance_content(self, text: str, title: str = "Content") -> GeneratedContent:
        """Generate enhanced content using rules"""
        print("\n🔄 Generating enhanced content (rule-based)...")
        
        simplified = self.simplify(text)
        takeaways = self.generate_takeaways(text, 5)
        faq = self.generate_questions(text, 4)
        vocabulary = self.extract_vocabulary(text, 8)
        
        # Simple elaboration
        sentences = text.split('.')[:2]
        elaboration = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
        
        # Simple ELI5
        first_sent = text.split('.')[0] if '.' in text else text[:100]
        eli5 = f"This is about {first_sent.lower().strip()}."
        
        # Simple examples placeholder
        concepts = re.findall(r'\b([A-Z][a-z]+)\b', text)[:3]
        examples = [f"Example of {c}: Common usage in everyday context." for c in concepts]
        
        return GeneratedContent(
            original=text,
            simplified_explanation=simplified,
            key_takeaways=takeaways,
            elaboration=elaboration,
            eli5_explanation=eli5,
            examples=examples,
            faq=faq,
            vocabulary=vocabulary
        )


def get_content_enhancer(use_ai: bool = True) -> ContentEnhancer:
    """
    Get the appropriate content enhancer
    
    Args:
        use_ai: Whether to use AI models (requires transformers)
        
    Returns:
        ContentEnhancer or OfflineContentGenerator
    """
    if use_ai:
        try:
            import transformers
            return ContentEnhancer()
        except ImportError:
            print("⚠️ transformers not available, using rule-based generator")
            return OfflineContentGenerator()
    else:
        return OfflineContentGenerator()
