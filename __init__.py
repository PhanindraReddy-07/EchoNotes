"""
EchoNotes - Offline Speech-to-Document Pipeline
================================================

A multilingual, offline-first speech-to-document system with:
- Audio capture and intelligent preprocessing
- Offline speech recognition (Vosk)
- Speaker diarization
- Domain-specific NER (Meeting entities)
- Hybrid summarization
- Multi-format document generation
- Web interface and IoT (Raspberry Pi) deployment

Modules:
    - audio: Audio capture, processing, and enhancement
    - speech: ASR and speaker diarization (coming next)
    - nlp: NER, summarization, preprocessing (coming soon)
    - document: Output generation (coming soon)
    - api: FastAPI web interface (coming soon)

Usage:
    from echonotes.audio import AudioCapture, IntelligentAudioPreprocessor
    from echonotes.config import get_config
    
    # Load and process audio
    capture = AudioCapture()
    audio = capture.load_file("meeting.wav")
    
    preprocessor = IntelligentAudioPreprocessor()
    enhanced, report = preprocessor.process_pipeline(audio)
    
    print(f"Quality: {report.overall_score}/100")

Author: EchoNotes Team
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'EchoNotes Team'

# Import main components for convenience
from .config import get_config, set_config, EchoNotesConfig

__all__ = [
    'get_config',
    'set_config', 
    'EchoNotesConfig',
    '__version__',
]
