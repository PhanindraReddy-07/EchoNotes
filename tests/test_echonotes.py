"""
EchoNotes Test Suite
Run: pytest tests/ -v
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


class TestPreprocessor:
    """Test text preprocessing."""
    
    def test_clean_text_removes_fillers(self):
        from modules.nlp_processor import Preprocessor
        from config import Settings
        
        settings = Settings()
        preprocessor = Preprocessor(settings)
        
        text = "um so like the meeting was you know pretty good"
        cleaned = preprocessor.clean_text(text)
        
        assert "um" not in cleaned.lower()
        assert "like" not in cleaned.lower() or "like" in "liked"
        assert "you know" not in cleaned.lower()
    
    def test_tokenize_sentences(self):
        from modules.nlp_processor import Preprocessor
        from config import Settings
        
        settings = Settings()
        preprocessor = Preprocessor(settings)
        
        text = "First sentence. Second sentence! Third one?"
        sentences = preprocessor.tokenize_sentences(text)
        
        assert len(sentences) == 3


class TestEntityExtractor:
    """Test entity extraction."""
    
    def test_extract_dates(self):
        from modules.nlp_processor import EntityExtractor
        from config import Settings
        
        settings = Settings()
        extractor = EntityExtractor(settings)
        
        text = "The meeting is on Monday. Let's meet on 12/25/2024."
        entities = extractor.extract(text)
        
        assert len(entities['dates']) > 0
    
    def test_extract_action_items(self):
        from modules.nlp_processor import EntityExtractor
        from config import Settings
        
        settings = Settings()
        extractor = EntityExtractor(settings)
        
        text = "We need to finish the report. Please send the email by Friday."
        actions = extractor.extract_action_items(text)
        
        assert len(actions) > 0


class TestDocumentStructurer:
    """Test document structuring."""
    
    def test_detect_meeting_type(self):
        from modules.nlp_processor import DocumentStructurer
        from config import Settings
        
        settings = Settings()
        structurer = DocumentStructurer(settings)
        
        meeting_text = "Meeting agenda: discuss project timeline. Attendees: John, Mary. Action items to follow."
        doc_type = structurer._detect_document_type(meeting_text)
        
        assert doc_type == 'meeting'
    
    def test_structure_document(self):
        from modules.nlp_processor import DocumentStructurer
        from config import Settings
        
        settings = Settings()
        structurer = DocumentStructurer(settings)
        
        doc = structurer.structure(
            text="Test content here.",
            sentences=["Test content here."],
            entities={'persons': [], 'organizations': [], 'locations': [], 'dates': [], 'times': [], 'money': []},
            summary="Test summary."
        )
        
        assert 'metadata' in doc
        assert 'title' in doc
        assert 'summary' in doc


class TestSummarizer:
    """Test summarization."""
    
    def test_extractive_summarize(self):
        from modules.nlp_processor import Summarizer
        from config import Settings
        
        settings = Settings()
        summarizer = Summarizer(settings)
        
        long_text = " ".join([
            "This is the first important sentence about the project.",
            "Here is some additional context that may be useful.",
            "The main point is that we need to finish on time.",
            "There are several factors to consider in our decision.",
            "Finally, we should review everything next week.",
        ] * 3)
        
        summary = summarizer._extractive_summarize(long_text, num_sentences=3)
        
        assert len(summary) < len(long_text)
        assert len(summary) > 50


class TestMarkdownGenerator:
    """Test markdown generation."""
    
    def test_generate_markdown(self, tmp_path):
        from modules.document_generator import MarkdownGenerator
        from config import Settings
        
        settings = Settings()
        generator = MarkdownGenerator(settings)
        
        doc_data = {
            'title': 'Test Document',
            'metadata': {'type': 'general', 'generated_at': '2024-01-01', 'word_count': 100},
            'summary': 'This is a test summary.',
            'content': {'full_text': 'Test content.', 'sections': []},
            'entities': {'persons': ['John'], 'organizations': [], 'locations': [], 'dates': []},
            'action_items': ['Review document'],
            'key_points': ['Main point one']
        }
        
        output_path = tmp_path / "test.md"
        result = generator.generate(doc_data, output_path)
        
        assert result.exists()
        content = result.read_text()
        assert 'Test Document' in content
        assert 'Test summary' in content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
