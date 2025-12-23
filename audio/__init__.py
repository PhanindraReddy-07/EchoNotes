"""
EchoNotes Audio Module
======================

Components for audio acquisition, processing, and enhancement.

Classes:
    - AudioCapture: Multi-source audio input (file, microphone, stream)
    - AudioData: Container for audio samples with metadata
    - AudioProcessor: Signal processing, filtering, VAD
    - SpeechSegment: Represents a detected speech segment
    - IntelligentAudioPreprocessor: Advanced noise reduction and quality assessment
    - AudioQualityReport: Detailed quality analysis report
    - NoiseType: Enum for detected noise types

Example Usage:
    from echonotes.audio import AudioCapture, AudioProcessor, IntelligentAudioPreprocessor
    
    # Load audio file
    capture = AudioCapture(target_sample_rate=16000)
    audio = capture.load_file("meeting.wav")
    
    # Analyze and enhance
    preprocessor = IntelligentAudioPreprocessor()
    enhanced, report = preprocessor.process_pipeline(audio)
    
    print(f"Quality Score: {report.overall_score}/100")
    print(f"Noise Type: {report.noise_type.value}")
    
    # Detect speech segments
    processor = AudioProcessor()
    segments = processor.detect_voice_activity(enhanced)
"""

from .capture import AudioCapture, AudioData
from .processor import AudioProcessor, SpeechSegment
from .enhancer import (
    IntelligentAudioPreprocessor,
    AudioQualityReport,
    NoiseType
)

__all__ = [
    # Capture
    'AudioCapture',
    'AudioData',
    
    # Processing
    'AudioProcessor',
    'SpeechSegment',
    
    # Enhancement (NEW)
    'IntelligentAudioPreprocessor',
    'AudioQualityReport',
    'NoiseType',
]

__version__ = '1.0.0'
