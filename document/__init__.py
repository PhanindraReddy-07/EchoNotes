"""
EchoNotes Document Module
=========================

Smart document generation with structured sections:
- Executive Summary
- Key Sentences
- Key Concepts (with definitions)
- Study Questions (easy/medium/hard)
- Related Topics
- Action Items (for meetings)
- Complete Content

Supported formats: Markdown, HTML, PDF, DOCX, TXT

Usage:
    from echonotes.document import SmartDocumentGenerator
    
    generator = SmartDocumentGenerator()
    generator.generate(
        text=transcript,
        output_path="notes.md",
        title="Meeting Notes"
    )
    
    # Or from file
    generator.generate_from_file("transcript.txt", "notes.html", format="html")
"""

from .smart_generator import SmartDocumentGenerator
from .generator import DocumentGenerator, ContentAnalyzer, DocumentContent

# Import smart analyzer from nlp module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.smart_analyzer import (
    SmartAnalyzer,
    ContentAnalysis,
    ExtractedConcept,
    GeneratedQuestion
)

__all__ = [
    # New improved generator
    'SmartDocumentGenerator',
    'SmartAnalyzer',
    'ContentAnalysis',
    'ExtractedConcept',
    'GeneratedQuestion',
    
    # Legacy generator
    'DocumentGenerator',
    'ContentAnalyzer', 
    'DocumentContent',
]

__version__ = '2.0.0'
