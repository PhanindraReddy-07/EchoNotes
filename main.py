"""
EchoNotes - Offline Speech-to-Document System
Windows/VS Code Version

Usage:
    python main.py --input recording.wav --output notes.md
    python main.py --record --duration 60 --output meeting.md
    python main.py --stream
    python main.py --text transcript.txt --output summary.md
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings
from modules.speech_recognition import AudioCapture, AudioProcessor, Transcriber
from modules.nlp_processor import Preprocessor, Summarizer, EntityExtractor, DocumentStructurer
from modules.document_generator import MarkdownGenerator, TextGenerator, JsonGenerator


class EchoNotes:
    """Main pipeline orchestrator for speech-to-document processing."""
    
    def __init__(self, config_path: Path = None):
        self.settings = Settings(config_path)
        self._init_modules()
    
    def _init_modules(self):
        """Initialize all processing modules."""
        # Speech recognition
        self.audio_capture = AudioCapture(self.settings)
        self.audio_processor = AudioProcessor(self.settings)
        self.transcriber = Transcriber(self.settings)
        
        # NLP processing
        self.preprocessor = Preprocessor(self.settings)
        self.summarizer = Summarizer(self.settings)
        self.entity_extractor = EntityExtractor(self.settings)
        self.document_structurer = DocumentStructurer(self.settings)
        
        # Document generators
        self.generators = {
            '.md': MarkdownGenerator(self.settings),
            '.txt': TextGenerator(self.settings),
            '.json': JsonGenerator(self.settings),
        }
    
    def process_audio_file(self, input_path: Path, output_path: Path) -> Path:
        """Process an audio file and generate document."""
        print(f"Processing: {input_path}")
        
        # Load and process audio
        audio_data, sample_rate = self.audio_capture.load_file(input_path)
        processed_audio = self.audio_processor.process(audio_data, sample_rate)
        
        # Transcribe
        print("Transcribing audio...")
        transcript = self.transcriber.transcribe(processed_audio)
        
        if not transcript.strip():
            print("Warning: No speech detected in audio")
            return None
        
        # Process text and generate document
        return self._process_text_to_document(transcript, output_path)
    
    def record_and_process(self, duration: int, output_path: Path) -> Path:
        """Record audio from microphone and process."""
        print(f"Recording for {duration} seconds...")
        print("Speak now!")
        
        audio_data, sample_rate = self.audio_capture.record(duration)
        processed_audio = self.audio_processor.process(audio_data, sample_rate)
        
        print("Processing recording...")
        transcript = self.transcriber.transcribe(processed_audio)
        
        if not transcript.strip():
            print("Warning: No speech detected")
            return None
        
        return self._process_text_to_document(transcript, output_path)
    
    def stream_process(self):
        """Stream audio and process in real-time."""
        print("Starting stream mode (Ctrl+C to stop)...")
        print("Speak now!")
        
        try:
            for transcript_chunk in self.transcriber.stream_transcribe(
                self.audio_capture.stream()
            ):
                if transcript_chunk.strip():
                    # Process each chunk
                    cleaned = self.preprocessor.clean_text(transcript_chunk)
                    print(f"\n[Transcribed]: {cleaned}")
        except KeyboardInterrupt:
            print("\nStream stopped.")
    
    def process_text_file(self, input_path: Path, output_path: Path) -> Path:
        """Process a text file (existing transcript)."""
        print(f"Processing text: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self._process_text_to_document(text, output_path)
    
    def _process_text_to_document(self, text: str, output_path: Path) -> Path:
        """Process text through NLP pipeline and generate document."""
        # Preprocess
        print("Cleaning text...")
        cleaned_text = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.tokenize_sentences(cleaned_text)
        
        # Extract entities
        print("Extracting entities...")
        entities = self.entity_extractor.extract(cleaned_text)
        
        # Generate summary
        print("Generating summary...")
        summary = self.summarizer.summarize(cleaned_text)
        
        # Structure document
        print("Structuring document...")
        doc_data = self.document_structurer.structure(
            text=cleaned_text,
            sentences=sentences,
            entities=entities,
            summary=summary,
            metadata={
                'generated_at': datetime.now().isoformat(),
                'source': 'EchoNotes',
            }
        )
        
        # Generate output
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()
        
        if suffix not in self.generators:
            suffix = '.md'  # Default to markdown
            output_path = output_path.with_suffix('.md')
        
        generator = self.generators[suffix]
        output_file = generator.generate(doc_data, output_path)
        
        print(f"\nDocument saved: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='EchoNotes - Offline Speech-to-Document System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input meeting.wav --output notes.md
  python main.py --record --duration 60 --output recording.md
  python main.py --stream
  python main.py --text transcript.txt --output summary.md
        """
    )
    
    # Input modes (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=Path, help='Input audio file')
    input_group.add_argument('--record', '-r', action='store_true', help='Record from microphone')
    input_group.add_argument('--stream', '-s', action='store_true', help='Stream mode (real-time)')
    input_group.add_argument('--text', '-t', type=Path, help='Process text file')
    
    # Options
    parser.add_argument('--output', '-o', type=Path, default=Path('output/notes.md'),
                        help='Output file path (default: output/notes.md)')
    parser.add_argument('--duration', '-d', type=int, default=30,
                        help='Recording duration in seconds (default: 30)')
    parser.add_argument('--config', '-c', type=Path, help='Custom config file')
    parser.add_argument('--small', action='store_true', 
                        help='Use small/lightweight models')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize EchoNotes
    try:
        echonotes = EchoNotes(args.config)
    except Exception as e:
        print(f"Error initializing EchoNotes: {e}")
        print("\nMake sure you've run: python setup_models.py")
        sys.exit(1)
    
    # Process based on mode
    try:
        if args.input:
            if not args.input.exists():
                print(f"Error: File not found: {args.input}")
                sys.exit(1)
            echonotes.process_audio_file(args.input, args.output)
        
        elif args.record:
            echonotes.record_and_process(args.duration, args.output)
        
        elif args.stream:
            echonotes.stream_process()
        
        elif args.text:
            if not args.text.exists():
                print(f"Error: File not found: {args.text}")
                sys.exit(1)
            echonotes.process_text_file(args.text, args.output)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
