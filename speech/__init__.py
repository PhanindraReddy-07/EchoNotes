"""
EchoNotes Speech Module
=======================

Components for speech recognition and speaker diarization.

Classes:
    - Transcriber: Offline ASR using Vosk
    - Word: Single transcribed word with timestamp
    - Utterance: Speech segment (sentence/phrase)
    - TranscriptionResult: Complete transcription result
    
    - SpeakerDiarizer: Speaker identification
    - SpeakerSegment: Speaker segment with timestamps
    - DiarizationResult: Complete diarization result
    
    - ConfidenceScorer: Word-level confidence analysis
    - ConfidenceReport: Confidence analysis report
    
    - LongFormProcessor: Process hours-long audio

Example Usage:
    from echonotes.speech import Transcriber, SpeakerDiarizer, ConfidenceScorer
    
    # Transcribe audio
    transcriber = Transcriber(model_path="path/to/vosk-model")
    result = transcriber.transcribe(audio_data)
    print(result.text)
    
    # Identify speakers
    diarizer = SpeakerDiarizer()
    diarization = diarizer.diarize(audio_data)
    diarizer.assign_speakers_to_transcript(diarization, result)
    
    # Analyze confidence
    scorer = ConfidenceScorer()
    report = scorer.analyze(result)
    print(f"Quality: {report.quality_rating}")
"""

from .transcriber import (
    Transcriber,
    Word,
    Utterance,
    TranscriptionResult
)

from .diarizer import (
    SpeakerDiarizer,
    SpeakerSegment,
    DiarizationResult
)

from .confidence import (
    ConfidenceScorer,
    ConfidenceReport,
    ConfidenceSpan,
    format_confidence_report
)

from .long_form import (
    LongFormProcessor,
    ProcessingProgress
)

__all__ = [
    # Transcription
    'Transcriber',
    'Word',
    'Utterance',
    'TranscriptionResult',
    
    # Diarization
    'SpeakerDiarizer',
    'SpeakerSegment',
    'DiarizationResult',
    
    # Confidence
    'ConfidenceScorer',
    'ConfidenceReport',
    'ConfidenceSpan',
    'format_confidence_report',
    
    # Long-form processing
    'LongFormProcessor',
    'ProcessingProgress',
]

__version__ = '1.0.0'
