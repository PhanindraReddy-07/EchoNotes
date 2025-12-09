"""
Markdown Document Generator
Generates formatted Markdown documents from structured data.
"""

from pathlib import Path
from typing import Dict, Any


class MarkdownGenerator:
    """Generate Markdown documents."""
    
    def __init__(self, settings):
        self.settings = settings
    
    def generate(self, doc_data: Dict[str, Any], output_path: Path) -> Path:
        """
        Generate Markdown document from structured data.
        
        Args:
            doc_data: Structured document data
            output_path: Output file path
            
        Returns:
            Path to generated file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build markdown content
        lines = []
        
        # Title
        title = doc_data.get('title', 'Notes')
        lines.append(f"# {title}")
        lines.append("")
        
        # Metadata
        metadata = doc_data.get('metadata', {})
        if metadata:
            lines.append(f"*Generated: {metadata.get('generated_at', 'Unknown')}*")
            lines.append(f"*Type: {metadata.get('type', 'general').title()}*")
            lines.append(f"*Words: {metadata.get('word_count', 0)}*")
            lines.append("")
        
        # Summary
        summary = doc_data.get('summary')
        doc_config = self.settings.config.get('document', {})
        if summary and doc_config.get('include_summary', True):
            lines.append("## Summary")
            lines.append("")
            lines.append(summary)
            lines.append("")
        
        # Key Points
        key_points = doc_data.get('key_points', [])
        if key_points:
            lines.append("## Key Points")
            lines.append("")
            for point in key_points:
                lines.append(f"- {point}")
            lines.append("")
        
        # Action Items
        action_items = doc_data.get('action_items', [])
        if action_items:
            lines.append("## Action Items")
            lines.append("")
            for item in action_items:
                lines.append(f"- [ ] {item}")
            lines.append("")
        
        # Entities
        entities = doc_data.get('entities', {})
        if entities and doc_config.get('include_entities', True):
            has_entities = any(entities.values())
            if has_entities:
                lines.append("## Entities Mentioned")
                lines.append("")
                
                if entities.get('persons'):
                    lines.append(f"**People:** {', '.join(entities['persons'])}")
                if entities.get('organizations'):
                    lines.append(f"**Organizations:** {', '.join(entities['organizations'])}")
                if entities.get('locations'):
                    lines.append(f"**Locations:** {', '.join(entities['locations'])}")
                if entities.get('dates'):
                    lines.append(f"**Dates:** {', '.join(entities['dates'])}")
                lines.append("")
        
        # Full Content
        content = doc_data.get('content', {})
        sections = content.get('sections', [])
        
        if sections:
            lines.append("## Content")
            lines.append("")
            
            for section in sections:
                if len(sections) > 1:
                    lines.append(f"### {section.get('title', 'Section')}")
                    lines.append("")
                
                for sentence in section.get('content', []):
                    lines.append(sentence)
                lines.append("")
        elif content.get('full_text'):
            lines.append("## Content")
            lines.append("")
            lines.append(content['full_text'])
            lines.append("")
        
        # Write file
        markdown_content = '\n'.join(lines)
        output_path.write_text(markdown_content, encoding='utf-8')
        
        return output_path