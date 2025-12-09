"""
Plain Text Document Generator
Generates simple text documents.
"""

from pathlib import Path
from typing import Dict, Any


class TextGenerator:
    """Generate plain text documents."""
    
    def __init__(self, settings):
        self.settings = settings
    
    def generate(self, doc_data: Dict[str, Any], output_path: Path) -> Path:
        """Generate plain text document."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        
        # Title
        title = doc_data.get('title', 'Notes')
        lines.append(title.upper())
        lines.append("=" * len(title))
        lines.append("")
        
        # Metadata
        metadata = doc_data.get('metadata', {})
        lines.append(f"Generated: {metadata.get('generated_at', 'Unknown')}")
        lines.append(f"Type: {metadata.get('type', 'general').title()}")
        lines.append("")
        
        # Summary
        summary = doc_data.get('summary')
        if summary:
            lines.append("SUMMARY")
            lines.append("-" * 7)
            lines.append(summary)
            lines.append("")
        
        # Key Points
        key_points = doc_data.get('key_points', [])
        if key_points:
            lines.append("KEY POINTS")
            lines.append("-" * 10)
            for i, point in enumerate(key_points, 1):
                lines.append(f"{i}. {point}")
            lines.append("")
        
        # Action Items
        action_items = doc_data.get('action_items', [])
        if action_items:
            lines.append("ACTION ITEMS")
            lines.append("-" * 12)
            for item in action_items:
                lines.append(f"[ ] {item}")
            lines.append("")
        
        # Content
        content = doc_data.get('content', {})
        full_text = content.get('full_text', '')
        
        if full_text:
            lines.append("CONTENT")
            lines.append("-" * 7)
            lines.append(full_text)
        
        # Write file
        text_content = '\n'.join(lines)
        output_path.write_text(text_content, encoding='utf-8')
        
        return output_path
