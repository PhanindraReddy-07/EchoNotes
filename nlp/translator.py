"""
Offline Translation Module for EchoNotes
=========================================
Provides translation capabilities for short-form content using
Helsinki-NLP models from Hugging Face.

Supports multiple Indian languages and common international languages.
Designed for offline operation after initial model download.

Usage:
    translator = OfflineTranslator()
    
    # Translate English to Hindi
    hindi_text = translator.translate("Hello world", target_lang="hi")
    
    # Auto-detect and translate to English
    english_text = translator.translate_to_english("नमस्ते दुनिया")
    
    # Get available languages
    languages = translator.get_supported_languages()
"""

import re
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Result of translation"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'original': self.original_text,
            'translated': self.translated_text,
            'source_lang': self.source_language,
            'target_lang': self.target_language,
            'confidence': self.confidence
        }


class OfflineTranslator:
    """
    Offline translation using Helsinki-NLP MarianMT models.
    
    These models are downloaded once and work completely offline.
    Optimized for short-form content (sentences, paragraphs).
    
    Supported language pairs:
    - English <-> Hindi
    - English <-> Tamil
    - English <-> Telugu
    - English <-> Bengali
    - English <-> Marathi
    - English <-> Gujarati
    - English <-> Kannada
    - English <-> Malayalam
    - English <-> Punjabi
    - English <-> Urdu
    - English <-> French
    - English <-> German
    - English <-> Spanish
    - English <-> Chinese
    - English <-> Japanese
    - English <-> Arabic
    """
    
    # Language code mappings
    LANGUAGE_CODES = {
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'bn': 'Bengali',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'ur': 'Urdu',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ar': 'Arabic',
        'ru': 'Russian',
        'pt': 'Portuguese',
        'it': 'Italian',
    }
    
    # Helsinki-NLP model mappings (source -> target)
    MODEL_MAPPINGS = {
        # English to Indian languages
        ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi',
        ('en', 'ta'): 'Helsinki-NLP/opus-mt-en-dra',  # Dravidian languages
        ('en', 'te'): 'Helsinki-NLP/opus-mt-en-dra',
        ('en', 'bn'): 'Helsinki-NLP/opus-mt-en-inc',  # Indic languages
        ('en', 'mr'): 'Helsinki-NLP/opus-mt-en-inc',
        ('en', 'gu'): 'Helsinki-NLP/opus-mt-en-inc',
        ('en', 'ur'): 'Helsinki-NLP/opus-mt-en-ur',
        
        # Indian languages to English
        ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en',
        ('ta', 'en'): 'Helsinki-NLP/opus-mt-dra-en',
        ('te', 'en'): 'Helsinki-NLP/opus-mt-dra-en',
        ('bn', 'en'): 'Helsinki-NLP/opus-mt-inc-en',
        ('mr', 'en'): 'Helsinki-NLP/opus-mt-inc-en',
        ('ur', 'en'): 'Helsinki-NLP/opus-mt-ur-en',
        
        # English to European languages
        ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
        ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
        ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
        ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it',
        ('en', 'pt'): 'Helsinki-NLP/opus-mt-en-pt',
        ('en', 'ru'): 'Helsinki-NLP/opus-mt-en-ru',
        
        # European languages to English
        ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
        ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
        ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
        ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en',
        ('pt', 'en'): 'Helsinki-NLP/opus-mt-pt-en',
        ('ru', 'en'): 'Helsinki-NLP/opus-mt-ru-en',
        
        # English to Asian languages
        ('en', 'zh'): 'Helsinki-NLP/opus-mt-en-zh',
        ('en', 'ja'): 'Helsinki-NLP/opus-mt-en-jap',
        ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar',
        
        # Asian languages to English
        ('zh', 'en'): 'Helsinki-NLP/opus-mt-zh-en',
        ('ja', 'en'): 'Helsinki-NLP/opus-mt-jap-en',
        ('ar', 'en'): 'Helsinki-NLP/opus-mt-ar-en',
    }
    
    # Unicode ranges for language detection
    LANGUAGE_UNICODE_RANGES = {
        'hi': [('\u0900', '\u097F')],  # Devanagari
        'ta': [('\u0B80', '\u0BFF')],  # Tamil
        'te': [('\u0C00', '\u0C7F')],  # Telugu
        'bn': [('\u0980', '\u09FF')],  # Bengali
        'gu': [('\u0A80', '\u0AFF')],  # Gujarati
        'kn': [('\u0C80', '\u0CFF')],  # Kannada
        'ml': [('\u0D00', '\u0D7F')],  # Malayalam
        'pa': [('\u0A00', '\u0A7F')],  # Gurmukhi (Punjabi)
        'ar': [('\u0600', '\u06FF')],  # Arabic
        'zh': [('\u4E00', '\u9FFF')],  # Chinese
        'ja': [('\u3040', '\u309F'), ('\u30A0', '\u30FF'), ('\u4E00', '\u9FFF')],  # Japanese
        'ru': [('\u0400', '\u04FF')],  # Cyrillic
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize translator
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "echonotes" / "translation"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._models = {}  # Cache loaded models
        self._tokenizers = {}
        self._available = None
    
    def _check_availability(self) -> bool:
        """Check if translation is available"""
        if self._available is not None:
            return self._available
        
        try:
            from transformers import MarianMTModel, MarianTokenizer
            self._available = True
        except ImportError:
            print("[Translator] Translation requires: pip install transformers sentencepiece")
            self._available = False
        
        return self._available
    
    def _load_model(self, source_lang: str, target_lang: str) -> Tuple:
        """Load translation model for language pair"""
        if not self._check_availability():
            return None, None
        
        key = (source_lang, target_lang)
        
        if key in self._models:
            return self._models[key], self._tokenizers[key]
        
        model_name = self.MODEL_MAPPINGS.get(key)
        if not model_name:
            print(f"[Translator] No model available for {source_lang} -> {target_lang}")
            return None, None
        
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            print(f"[Translator] Loading model: {model_name}")
            
            tokenizer = MarianTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            model = MarianMTModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            self._models[key] = model
            self._tokenizers[key] = tokenizer
            
            print(f"[Translator] Model loaded for {source_lang} -> {target_lang}")
            return model, tokenizer
            
        except Exception as e:
            print(f"[Translator] Error loading model: {e}")
            return None, None
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text using Unicode character ranges
        
        Args:
            text: Text to detect language of
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'hi', 'ta')
        """
        if not text:
            return 'en'
        
        # Count characters in each language range
        lang_counts = {lang: 0 for lang in self.LANGUAGE_UNICODE_RANGES}
        latin_count = 0
        total_alpha = 0
        
        for char in text:
            if char.isalpha():
                total_alpha += 1
                
                # Check Latin (English and European)
                if '\u0041' <= char <= '\u007A':
                    latin_count += 1
                    continue
                
                # Check other scripts
                for lang, ranges in self.LANGUAGE_UNICODE_RANGES.items():
                    for start, end in ranges:
                        if start <= char <= end:
                            lang_counts[lang] += 1
                            break
        
        if total_alpha == 0:
            return 'en'
        
        # Find dominant language
        max_count = latin_count
        detected = 'en'
        
        for lang, count in lang_counts.items():
            if count > max_count:
                max_count = count
                detected = lang
        
        return detected
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        return self.LANGUAGE_CODES.copy()
    
    def get_available_pairs(self) -> List[Tuple[str, str]]:
        """Get list of available translation pairs"""
        return list(self.MODEL_MAPPINGS.keys())
    
    def is_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """Check if translation pair is supported"""
        return (source_lang, target_lang) in self.MODEL_MAPPINGS
    
    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None
    ) -> TranslationResult:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_lang: Target language code (e.g., 'hi', 'en')
            source_lang: Source language (auto-detected if not provided)
            
        Returns:
            TranslationResult with original and translated text
        """
        if not text or not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language='unknown',
                target_language=target_lang,
                confidence=0.0
            )
        
        # Auto-detect source language if not provided
        if source_lang is None:
            source_lang = self.detect_language(text)
        
        # If same language, return as-is
        if source_lang == target_lang:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=1.0
            )
        
        # Check if pair is supported
        if not self.is_pair_supported(source_lang, target_lang):
            # Try via English as pivot
            if source_lang != 'en' and target_lang != 'en':
                if self.is_pair_supported(source_lang, 'en') and self.is_pair_supported('en', target_lang):
                    # Translate to English first
                    intermediate = self.translate(text, 'en', source_lang)
                    # Then to target
                    final = self.translate(intermediate.translated_text, target_lang, 'en')
                    return TranslationResult(
                        original_text=text,
                        translated_text=final.translated_text,
                        source_language=source_lang,
                        target_language=target_lang,
                        confidence=intermediate.confidence * final.confidence
                    )
            
            return TranslationResult(
                original_text=text,
                translated_text=f"[Translation not available for {source_lang} -> {target_lang}]",
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.0
            )
        
        # Load model
        model, tokenizer = self._load_model(source_lang, target_lang)
        if model is None:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.0
            )
        
        try:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate translation
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            
            # Decode
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return TranslationResult(
                original_text=text,
                translated_text=translated,
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.9
            )
            
        except Exception as e:
            print(f"[Translator] Translation error: {e}")
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.0
            )
    
    def translate_to_english(self, text: str) -> TranslationResult:
        """
        Translate text to English (auto-detect source language)
        
        Args:
            text: Text in any supported language
            
        Returns:
            TranslationResult with English translation
        """
        return self.translate(text, target_lang='en')
    
    def translate_from_english(self, text: str, target_lang: str) -> TranslationResult:
        """
        Translate English text to target language
        
        Args:
            text: English text
            target_lang: Target language code
            
        Returns:
            TranslationResult with translation
        """
        return self.translate(text, target_lang=target_lang, source_lang='en')
    
    def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: Optional[str] = None
    ) -> List[TranslationResult]:
        """
        Translate multiple texts
        
        Args:
            texts: List of texts to translate
            target_lang: Target language
            source_lang: Source language (auto-detected if not provided)
            
        Returns:
            List of TranslationResults
        """
        return [self.translate(text, target_lang, source_lang) for text in texts]


# Convenience function
def get_translator() -> OfflineTranslator:
    """Get translator instance"""
    return OfflineTranslator()


# Language detection helper
def detect_language(text: str) -> str:
    """Detect language of text"""
    translator = OfflineTranslator()
    return translator.detect_language(text)
