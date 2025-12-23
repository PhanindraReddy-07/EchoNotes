#!/usr/bin/env python3
"""
EchoNotes Module 4 Demo - Smart Document Generation
====================================================

Generates structured documents with:
- Executive Summary
- Key Sentences
- Key Concepts
- Study Questions
- Related Topics
- Complete Content

Usage:
    python demo_document.py                           # Use sample text
    python demo_document.py transcript.txt            # From transcript file
    python demo_document.py transcript.txt --format pdf   # Generate PDF
    python demo_document.py transcript.txt --format html  # Generate HTML
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from document.smart_generator import SmartDocumentGenerator
from nlp.smart_analyzer import SmartAnalyzer


# Sample content for demo
SAMPLE_CONTENT = """
Instagram is an American photo and video sharing social networking service owned by Meta Platforms. 
It allows users to upload media that can be edited with filters and organized with tags and location information.

Posts can be shared publicly or with preapproved followers. Users can browse other users content by tags and locations, 
view trending content, like photos, and follow other users to add their content to a personal feed.

Instagram was originally created by Kevin Systrom and Mike Krieger. It was launched in October 2010 on iOS. 
The Android version was released in April 2012, followed by a desktop interface in November 2012.

The app allows users to upload photos and short videos, follow other users feeds, add geotag to their photos, 
connect their Instagram account to other social networking sites, and engage through likes and comments.

Instagram has become one of the most popular social media platforms with over 2 billion monthly active users. 
It has evolved to include features like Stories, Reels, and Shopping. The platform is particularly popular among 
younger demographics and has significant influence on digital marketing and influencer culture.
"""


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="EchoNotes Smart Document Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_document.py                           # Generate from sample
  python demo_document.py recording_transcript.txt  # From transcript
  python demo_document.py transcript.txt -f pdf     # Generate PDF
  python demo_document.py transcript.txt -f html    # Generate HTML
  python demo_document.py transcript.txt -f docx    # Generate Word doc
  python demo_document.py transcript.txt -o notes   # Custom output name
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input transcript file')
    parser.add_argument('-f', '--format', default='md', 
                        choices=['md', 'html', 'pdf', 'docx', 'txt'],
                        help='Output format (default: md)')
    parser.add_argument('-o', '--output', help='Output filename (without extension)')
    parser.add_argument('-t', '--title', default='EchoNotes Document',
                        help='Document title')
    parser.add_argument('--ai', action='store_true',
                        help='Enable AI-generated content (requires: pip install transformers torch)')
    parser.add_argument('--no-ai', action='store_true',
                        help='Use rule-based content generation (no AI, works offline)')
    
    args = parser.parse_args()
    
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
    ║     N O T E S   -   Module 4: Document Generation        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Load input text
    if args.input and Path(args.input).exists():
        print(f"📄 Loading: {args.input}")
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
        title = args.title if args.title != 'EchoNotes Document' else Path(args.input).stem.replace('_', ' ').title()
    else:
        if args.input:
            print(f"⚠️  File not found: {args.input}")
        print("📄 Using sample content for demo")
        text = SAMPLE_CONTENT
        title = args.title if args.title != 'EchoNotes Document' else "Instagram Overview"
    
    # Determine output path
    if args.output:
        output_name = args.output
    elif args.input:
        output_name = Path(args.input).stem + "_notes"
    else:
        output_name = "echonotes_output"
    
    output_path = f"{output_name}.{args.format}"
    
    print_header("ANALYZING CONTENT")
    
    # Generate document with improved analyzer
    generator = SmartDocumentGenerator()
    
    print(f"\n🔍 Analyzing content...")
    print(f"   Title: {title}")
    print(f"   Format: {args.format.upper()}")
    
    # Analyze first to show preview
    analyzer = SmartAnalyzer()
    analysis = analyzer.analyze(text, title)
    
    print(f"\n📊 Content Analysis:")
    print(f"   Words: {analysis.word_count}")
    print(f"   Sentences: {analysis.sentence_count}")
    print(f"   Reading Time: {analysis.reading_time_minutes} min")
    print(f"   Key Concepts: {len(analysis.concepts)}")
    print(f"   Study Questions: {len(analysis.questions)}")
    print(f"   Related Topics: {len(analysis.related_topics)}")
    
    print_header("EXECUTIVE SUMMARY")
    print(f"\n{analysis.executive_summary}")
    
    print_header("KEY SENTENCES")
    for i, sent in enumerate(analysis.key_sentences[:5], 1):
        preview = sent[:80] + "..." if len(sent) > 80 else sent
        print(f"  {i}. {preview}")
    
    print_header("KEY CONCEPTS")
    for concept in analysis.concepts[:5]:
        stars = "⭐" * int(concept.importance_score * 5)
        print(f"  • {concept.term} {stars}")
        print(f"    └─ {concept.definition[:60]}...")
    
    print_header("STUDY QUESTIONS")
    for i, q in enumerate(analysis.questions[:5], 1):
        emoji = {'easy': '🟢', 'medium': '🟡', 'hard': '🔴'}.get(q.difficulty, '⚪')
        print(f"  {i}. {emoji} [{q.question_type}] {q.question}")
    
    print_header("RELATED TOPICS")
    if analysis.related_topics:
        print(f"  {', '.join(analysis.related_topics)}")
    else:
        print("  (No specific topics identified)")
    
    # Show meeting items if found
    if analysis.action_items:
        print_header("ACTION ITEMS DETECTED")
        for item in analysis.action_items:
            print(f"  ☐ {item}")
    
    # Determine AI mode
    use_ai = args.ai and not args.no_ai
    
    if use_ai:
        print("\n🤖 AI Mode: ENABLED")
        print("   Using Flan-T5 for content generation")
    else:
        print("\n📝 AI Mode: DISABLED (rule-based generation)")
        print("   Enable with: --ai flag")
    
    # Generate file
    print_header("GENERATING DOCUMENT")
    
    output_file = generator.generate(
        text=text,
        output_path=output_path,
        title=title,
        format=args.format,
        use_ai=use_ai
    )
    
    print(f"\n✅ Document generated successfully!")
    print(f"   📁 File: {output_file}")
    print(f"   📊 Format: {args.format.upper()}")
    
    # Show file size
    size = Path(output_file).stat().st_size
    if size < 1024:
        size_str = f"{size} bytes"
    else:
        size_str = f"{size/1024:.1f} KB"
    print(f"   💾 Size: {size_str}")
    
    print_header("DEMO COMPLETE")
    print("""
  Document Sections Generated:
  
  ✅ Executive Summary
     - Concise overview of main points
     
  ✅ Key Sentences  
     - Most important sentences extracted
     
  ✅ Key Concepts
     - Important terms with definitions
     - Importance rating (⭐)
     
  ✅ Study Questions
     - Factual questions (🟢 Easy)
     - Conceptual questions (🟡 Medium)
     - Analysis questions (🔴 Hard)
     
  ✅ Related Topics
     - Suggested topics for further study
  
  🤖 AI-Enhanced Sections (with --ai flag):
     
  ✅ Simple Explanation
     - Easier to understand version
     
  ✅ ELI5 (Explain Like I'm 5)
     - Very simple explanation
     
  ✅ Key Takeaways
     - AI-generated main points
     
  ✅ Real-World Examples
     - Practical examples of concepts
     
  ✅ FAQ
     - Auto-generated Q&A pairs
     
  ✅ Vocabulary
     - Key terms with definitions
  
  Supported Formats:
     • Markdown (.md)  - Best for GitHub/notes
     • HTML (.html)    - Beautiful styled document
     • PDF (.pdf)      - Requires: pip install fpdf2
     • Word (.docx)    - Requires: pip install python-docx
     • Text (.txt)     - Plain text fallback
     
  AI Requirements (for --ai flag):
     pip install transformers torch
     
  Models used (downloaded automatically, ~100MB):
     • google/flan-t5-small (77MB) - Content generation
    """)


if __name__ == "__main__":
    main()
