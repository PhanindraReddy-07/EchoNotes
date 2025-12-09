"""
Summarizer Module
Transformer-based summarization with extractive fallback.
"""

import re
from typing import Optional

# Try to import transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class Summarizer:
    """Text summarization using transformers or extractive methods."""
    
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.summarizer_pipeline = None
        
        nlp_config = settings.config.get('nlp', {})
        self.max_length = nlp_config.get('max_summary_length', 150) or 150
        self.min_length = nlp_config.get('min_summary_length', 50) or 50
        self.use_extractive = nlp_config.get('use_extractive_fallback', True)
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load summarization model."""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        # Choose model based on settings
        nlp_config = self.settings.config.get('nlp', {})
        performance_config = self.settings.config.get('performance', {})
        
        low_memory = performance_config.get('low_memory_mode', False)
        
        if low_memory:
            model_name = nlp_config.get('summarizer_model_small', 'google/flan-t5-small')
        else:
            model_name = nlp_config.get('summarizer_model', 'sshleifer/distilbart-cnn-12-6')
        
        # Check if model is cached
        models_dir = self.settings.models_dir / 'summarizer'
        
        try:
            if models_dir.exists():
                # Load from local cache
                self.summarizer_pipeline = pipeline(
                    'summarization',
                    model=str(models_dir),
                    device=-1  # CPU
                )
                print(f"Loaded summarizer from cache: {models_dir.name}")
            else:
                # Try to load (will use HuggingFace cache)
                use_gpu = nlp_config.get('use_gpu', False)
                device = 0 if (use_gpu and torch.cuda.is_available()) else -1
                self.summarizer_pipeline = pipeline(
                    'summarization',
                    model=model_name,
                    device=device
                )
                print(f"Loaded summarizer: {model_name}")
        except Exception as e:
            print(f"Warning: Could not load summarizer model: {e}")
            print("Will use extractive summarization.")
    
    def summarize(self, text: str) -> str:
        """
        Generate summary of text.
        
        Uses transformer model if available, otherwise extractive method.
        """
        if not text or len(text.split()) < 30:
            return text  # Too short to summarize
        
        # Try transformer summarization
        if self.summarizer_pipeline:
            try:
                return self._transformer_summarize(text)
            except Exception as e:
                print(f"Transformer summarization failed: {e}")
        
        # Fallback to extractive
        if self.use_extractive:
            return self._extractive_summarize(text)
        
        return text[:500] + "..." if len(text) > 500 else text
    
    def _transformer_summarize(self, text: str) -> str:
        """Summarize using transformer model."""
        # Truncate input if too long
        max_input = 1024
        words = text.split()
        if len(words) > max_input:
            text = ' '.join(words[:max_input])
        
        result = self.summarizer_pipeline(
            text,
            max_length=self.max_length,
            min_length=self.min_length,
            do_sample=False,
            truncation=True
        )
        
        return result[0]['summary_text']
    
    def _extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Simple extractive summarization.
        Selects most important sentences based on word frequency.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate word frequencies (simple TF)
        words = text.lower().split()
        word_freq = {}
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'to', 'of', 'in', 'for', 'on', 'with',
                     'at', 'by', 'from', 'as', 'and', 'or', 'but', 'this', 'that'}
        
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if word and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            sent_words = sentence.lower().split()
            for word in sent_words:
                word = re.sub(r'[^\w]', '', word)
                score += word_freq.get(word, 0)
            
            # Normalize by sentence length
            if sent_words:
                score = score / len(sent_words)
            
            # Boost first and last sentences slightly
            if i == 0:
                score *= 1.2
            elif i == len(sentences) - 1:
                score *= 1.1
            
            sentence_scores.append((i, score, sentence))
        
        # Select top sentences
        top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Return in original order
        top_sentences.sort(key=lambda x: x[0])
        
        return ' '.join(s[2] for s in top_sentences)
    
    @property
    def is_available(self) -> bool:
        """Check if summarizer is ready."""
        return self.summarizer_pipeline is not None or self.use_extractive