#!/usr/bin/env python3
"""
EchoNotes Module 3 Demo - Advanced NLP Processing
==================================================

Demonstrates:
1. Text Preprocessing (filler removal, cleaning)
2. Domain-Specific Entity Extraction (actions, decisions, deadlines)
3. Code-Mixed Language Handling (Hindi-English, Telugu-English)
4. Hybrid Summarization (transformer + extractive)

Usage:
    python demo_nlp.py                    # Run with sample meeting text
    python demo_nlp.py transcript.txt     # Run with your transcript file
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nlp import (
    TextPreprocessor,
    MeetingEntityExtractor,
    CodeMixHandler,
    HybridSummarizer,
    Language
)


# Sample meeting transcript for demo
SAMPLE_MEETING_TRANSCRIPT = """
Okay so um basically we need to discuss the Q4 roadmap today. 
John will prepare the technical specification document by Friday.
Sarah needs to review the budget proposals and get back to us by next week.

We decided to use Python for the backend services instead of Java.
The deadline for the MVP is December 15th. This is like really important you know.

There's a risk that the third-party API might not be ready on time.
We're blocked by the security team approval. Mike is waiting for their sign-off.

What about the mobile app timeline? Can someone look into that?
Tom should coordinate with the design team about the new UI mockups.

I think we need to be careful about the database migration. 
If it fails, we could lose customer data. That's a major concern.

Action item: Lisa will set up the staging environment by end of day tomorrow.
We agreed that weekly standups will happen on Mondays at 10 AM.

The client wants the demo ready by January 5th, so we have to finish testing by December 30th.
Let's schedule a follow-up meeting next Tuesday to track progress.
"""

# Sample code-mixed text (Hindi-English)
SAMPLE_HINDI_ENGLISH = """
Meeting kal postpone kar diya hai. John ne report bhej di.
Hume deadline se pehle complete karna hai. Kya aap review kar sakte ho?
Backend team se baat karo aur update do. Sprint planning next week hai.
"""

# Sample code-mixed text (Telugu-English)
SAMPLE_TELUGU_ENGLISH = """
Meeting repu postpone chesaru. Nenu report prepare chesthanu by Friday.
Deadline varaku complete cheyali. Mee review kavali please.
Backend team tho discuss chesi update ivvandi. Sprint ki ready avvali.
"""


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def demo_preprocessing(text: str):
    """Demo text preprocessing"""
    print_header("1. TEXT PREPROCESSING")
    
    preprocessor = TextPreprocessor(
        remove_fillers=True,
        include_indian_fillers=True
    )
    
    result = preprocessor.process(text)
    
    print(f"\n📊 Statistics:")
    print(f"   Original length: {len(result.original)} chars")
    print(f"   Cleaned length:  {len(result.cleaned)} chars")
    print(f"   Sentences:       {len(result.sentences)}")
    print(f"   Word count:      {result.word_count}")
    print(f"   Fillers removed: {result.filler_count}")
    
    if result.fillers_removed:
        print(f"\n🗑️  Fillers removed: {result.fillers_removed[:10]}")
    
    print(f"\n📝 Cleaned text (first 500 chars):")
    print(f"   {result.cleaned[:500]}...")
    
    return result.cleaned


def demo_entity_extraction(text: str):
    """Demo domain-specific entity extraction"""
    print_header("2. DOMAIN-SPECIFIC ENTITY EXTRACTION")
    
    extractor = MeetingEntityExtractor()
    result = extractor.extract(text)
    
    print(f"\n📊 Entities Found:")
    print(f"   Action Items: {len(result.action_items)}")
    print(f"   Deadlines:    {len(result.deadlines)}")
    print(f"   Decisions:    {len(result.decisions)}")
    print(f"   Questions:    {len(result.questions)}")
    print(f"   Risks:        {len(result.risks)}")
    print(f"   Blockers:     {len(result.blockers)}")
    
    # Show formatted summary
    print(result.get_summary())
    
    return result


def demo_code_mix_hindi(text: str):
    """Demo Hindi-English code-mix handling"""
    print_header("3a. CODE-MIX: HINDI-ENGLISH")
    
    handler = CodeMixHandler()
    result = handler.analyze(text)
    
    print(f"\n🔍 Analysis:")
    print(f"   Primary Language: {result.primary_language.value}")
    print(f"   Is Code-Mixed:    {result.is_code_mixed}")
    print(f"\n📊 Language Distribution:")
    for lang, pct in result.language_distribution.items():
        bar = "█" * int(pct / 5)
        print(f"   {lang}: {pct:>5.1f}% {bar}")
    
    print(f"\n📝 Bilingual Format:")
    print(f"   {handler.format_bilingual(result)}")
    
    return result


def demo_code_mix_telugu(text: str):
    """Demo Telugu-English code-mix handling"""
    print_header("3b. CODE-MIX: TELUGU-ENGLISH")
    
    handler = CodeMixHandler()
    result = handler.analyze(text)
    
    print(f"\n🔍 Analysis:")
    print(f"   Primary Language: {result.primary_language.value}")
    print(f"   Is Code-Mixed:    {result.is_code_mixed}")
    print(f"\n📊 Language Distribution:")
    for lang, pct in result.language_distribution.items():
        bar = "█" * int(pct / 5)
        print(f"   {lang}: {pct:>5.1f}% {bar}")
    
    print(f"\n📝 Segments:")
    for seg in result.segments[:5]:
        print(f"   [{seg.language.value}] {seg.text}")
    
    return result


def demo_summarization(text: str):
    """Demo hybrid summarization"""
    print_header("4. HYBRID SUMMARIZATION")
    
    # Without transformer (no heavy dependencies)
    summarizer = HybridSummarizer(use_transformer=False)
    
    print("\n🔄 Generating summary...")
    
    # Auto strategy
    result = summarizer.summarize(text, strategy="auto", is_meeting=True)
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Strategy Used:     {result.strategy_used}")
    print(f"   Confidence:        {result.confidence:.1%}")
    print(f"   Word Count:        {result.word_count}")
    print(f"   Compression Ratio: {result.compression_ratio:.1%}")
    
    print(f"\n📝 SUMMARY:")
    print("-" * 50)
    print(result.summary)
    print("-" * 50)
    
    if result.key_points:
        print(f"\n🔑 Key Points:")
        for i, point in enumerate(result.key_points, 1):
            print(f"   {i}. {point[:80]}{'...' if len(point) > 80 else ''}")
    
    # Also try ensemble
    print("\n\n🔄 Trying ENSEMBLE strategy...")
    ensemble_result = summarizer.summarize(text, strategy="ensemble", is_meeting=True)
    
    print(f"\n📝 ENSEMBLE SUMMARY:")
    print("-" * 50)
    print(ensemble_result.summary)
    print("-" * 50)
    
    return result


def demo_with_transformer(text: str):
    """Demo with transformer summarization (requires transformers library)"""
    print_header("4b. TRANSFORMER SUMMARIZATION (Optional)")
    
    try:
        summarizer = HybridSummarizer(use_transformer=True)
        result = summarizer.summarize(text, strategy="transformer")
        
        print(f"\n📝 TRANSFORMER SUMMARY:")
        print("-" * 50)
        print(result.summary)
        print("-" * 50)
        print(f"\n   Confidence: {result.confidence:.1%}")
        
    except Exception as e:
        print(f"\n⚠️  Transformer not available: {e}")
        print("   Install with: pip install transformers torch")
        print("   This is optional - extractive summarization works without it.")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ███████╗ ██████╗██╗  ██╗ ██████╗                     ║
    ║     ██╔════╝██╔════╝██║  ██║██╔═══██╗                    ║
    ║     █████╗  ██║     ███████║██║   ██║                    ║
    ║     ██╔══╝  ██║     ██╔══██║██║   ██║                    ║
    ║     ███████╗╚██████╗██║  ██║╚██████╔╝                    ║
    ║     ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝                    ║
    ║                                                           ║
    ║     N O T E S   -   Module 3: NLP Processing             ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check for custom transcript file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if Path(filepath).exists():
            print(f"📄 Loading transcript from: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            print(f"❌ File not found: {filepath}")
            print("   Using sample transcript instead.")
            text = SAMPLE_MEETING_TRANSCRIPT
    else:
        print("📄 Using sample meeting transcript")
        print("   (Run with: python demo_nlp.py your_transcript.txt)")
        text = SAMPLE_MEETING_TRANSCRIPT
    
    # Run demos
    cleaned_text = demo_preprocessing(text)
    entities = demo_entity_extraction(cleaned_text)
    
    # Code-mix demos
    demo_code_mix_hindi(SAMPLE_HINDI_ENGLISH)
    demo_code_mix_telugu(SAMPLE_TELUGU_ENGLISH)
    
    # Summarization
    summary = demo_summarization(cleaned_text)
    
    # Optional transformer demo
    # demo_with_transformer(cleaned_text)
    
    # Final summary
    print_header("DEMO COMPLETE")
    print("""
  Module 3 Components Demonstrated:
  
  ✅ TextPreprocessor
     • Filler word removal (um, uh, basically, etc.)
     • Indian English fillers (actually, basically, na, etc.)
     • Sentence segmentation
     
  ✅ MeetingEntityExtractor (NOVEL)
     • ACTION_ITEM detection with assignee
     • DEADLINE extraction
     • DECISION identification
     • QUESTION detection
     • RISK identification
     • BLOCKER detection
     
  ✅ CodeMixHandler (NOVEL)
     • Hindi-English code-mixing
     • Telugu-English code-mixing
     • Word-level language detection
     • Technical term preservation
     
  ✅ HybridSummarizer (NOVEL)
     • TextRank extractive
     • Clustering-based extractive
     • Meeting pattern extraction
     • Ensemble combination
     • (Optional) Transformer abstractive

  Technical Novelty Claims:
  1. Domain-specific NER for meeting entities
  2. Indian language code-mix handling
  3. Hybrid summarization reducing hallucination
  
  Next: Module 4 (Document Generation) for:
     • Markdown/PDF/DOCX output
     • Template-based formatting
     • Export to Google Docs/Notion
    """)


if __name__ == "__main__":
    main()
