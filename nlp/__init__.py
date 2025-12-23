"""
EchoNotes NLP Module
====================

Advanced NLP processing for meeting transcripts featuring:
- Domain-specific entity extraction (actions, decisions, deadlines)
- Code-mixed language handling (Telugu-English, Hindi-English)
- Hybrid summarization (transformer + extractive)
- Text preprocessing

TECHNICAL NOVELTY:
1. Custom NER for meeting entities (not in generic SpaCy/NLTK)
2. Code-mix detection for Indian languages
3. Ensemble summarization reducing hallucination

Classes:
    - TextPreprocessor: Clean transcripts, remove fillers
    - MeetingEntityExtractor: Domain-specific NER
    - CodeMixHandler: Telugu/Hindi-English detection
    - HybridSummarizer: Multi-strategy summarization

Example Usage:
    from echonotes.nlp import (
        TextPreprocessor,
        MeetingEntityExtractor,
        CodeMixHandler,
        HybridSummarizer
    )
    
    # Preprocess
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.process(transcript)
    
    # Extract meeting entities
    extractor = MeetingEntityExtractor()
    entities = extractor.extract(cleaned.cleaned)
    print(entities.get_summary())
    
    # Handle code-mixed text
    handler = CodeMixHandler()
    result = handler.analyze("Meeting kal postpone kar diya")
    print(f"Languages: {result.language_distribution}")
    
    # Summarize
    summarizer = HybridSummarizer()
    summary = summarizer.summarize(cleaned.cleaned)
    print(summary.summary)
"""

from .preprocessor import (
    TextPreprocessor,
    ProcessedText
)

from .entity_extractor import (
    MeetingEntityExtractor,
    MeetingEntity,
    MeetingEntityType,
    ExtractionResult
)

from .code_mix import (
    CodeMixHandler,
    CodeMixResult,
    LanguageSegment,
    Language,
    HINDI_ENGLISH_PHRASES,
    TELUGU_ENGLISH_PHRASES
)

from .summarizer import (
    HybridSummarizer,
    SummaryResult,
    TextRankSummarizer,
    ClusteringSummarizer,
    TransformerSummarizer,
    MeetingPatternSummarizer
)

__all__ = [
    # Preprocessing
    'TextPreprocessor',
    'ProcessedText',
    
    # Entity Extraction (NEW - Domain-specific)
    'MeetingEntityExtractor',
    'MeetingEntity',
    'MeetingEntityType',
    'ExtractionResult',
    
    # Code-Mix Handling (NEW - Indian languages)
    'CodeMixHandler',
    'CodeMixResult',
    'LanguageSegment',
    'Language',
    'HINDI_ENGLISH_PHRASES',
    'TELUGU_ENGLISH_PHRASES',
    
    # Summarization (NEW - Hybrid approach)
    'HybridSummarizer',
    'SummaryResult',
    'TextRankSummarizer',
    'ClusteringSummarizer',
    'TransformerSummarizer',
    'MeetingPatternSummarizer',
]

__version__ = '1.0.0'
