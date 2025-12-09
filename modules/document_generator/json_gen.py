"""
JSON Document Generator
Generates structured JSON output.
"""

import json
from pathlib import Path
from typing import Dict, Any


class JsonGenerator:
    """Generate JSON documents."""
    
    def __init__(self, settings):
        self.settings = settings
    
    def generate(self, doc_data: Dict[str, Any], output_path: Path) -> Path:
        """Generate JSON document."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write formatted JSON
        json_content = json.dumps(doc_data, indent=2, ensure_ascii=False)
        output_path.write_text(json_content, encoding='utf-8')
        
        return output_path
