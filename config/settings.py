"""
EchoNotes Configuration Settings
Central configuration for all modules
"""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import os

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 30.0  # seconds per chunk for long-form processing
    min_audio_length: float = 0.5  # minimum audio length in seconds
    max_audio_length: float = 14400.0  # 4 hours max
    supported_formats: List[str] = field(default_factory=lambda: ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm'])
    
    # VAD settings
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    vad_frame_duration: int = 30  # ms
    min_speech_duration: float = 0.3  # seconds
    
    # Audio enhancement
    noise_reduce_strength: float = 0.7
    normalize_audio: bool = True
    target_db: float = -20.0

@dataclass
class ASRConfig:
    """Speech recognition configuration"""
    model_path: Optional[str] = None  # Vosk model path
    model_name: str = "vosk-model-en-us-0.22"  # Default English model
    small_model_name: str = "vosk-model-small-en-us-0.15"  # For Raspberry Pi
    use_small_model: bool = False  # Toggle for resource-constrained devices
    
    # Streaming settings
    buffer_size: int = 4000
    max_alternatives: int = 3
    words: bool = True  # Get word-level timestamps
    
    # Confidence thresholds
    min_confidence: float = 0.5
    high_confidence: float = 0.85

@dataclass
class DiarizationConfig:
    """Speaker diarization configuration"""
    min_speakers: int = 1
    max_speakers: int = 10
    embedding_model: str = "resemblyzer"  # or "speechbrain"
    clustering_method: str = "agglomerative"  # or "spectral"
    min_cluster_size: int = 2
    
    # Segment settings
    min_segment_duration: float = 0.5
    merge_threshold: float = 0.5  # seconds between same-speaker segments

@dataclass
class NLPConfig:
    """NLP processing configuration"""
    # SpaCy settings
    spacy_model: str = "en_core_web_sm"
    
    # Custom NER
    custom_ner_model_path: Optional[str] = None
    entity_types: List[str] = field(default_factory=lambda: [
        'ACTION_ITEM', 'DEADLINE', 'DECISION', 'QUESTION', 'RISK', 'BLOCKER'
    ])
    
    # Summarization
    summarizer_model: str = "facebook/bart-large-cnn"
    small_summarizer_model: str = "sshleifer/distilbart-cnn-12-6"
    max_summary_length: int = 150
    min_summary_length: int = 30
    
    # Preprocessing
    remove_fillers: bool = True
    filler_words: List[str] = field(default_factory=lambda: [
        'um', 'uh', 'er', 'ah', 'like', 'you know', 'basically', 'actually',
        'literally', 'honestly', 'so', 'well', 'right', 'okay', 'i mean'
    ])

@dataclass
class OutputConfig:
    """Document output configuration"""
    output_dir: str = "./outputs"
    formats: List[str] = field(default_factory=lambda: ['md', 'pdf', 'docx'])
    include_timestamps: bool = True
    include_speakers: bool = True
    include_confidence: bool = False  # Show confidence scores
    
    # Template settings
    template_dir: str = "./templates"
    default_template: str = "meeting"  # meeting, lecture, interview

@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Device settings
    device: str = "cpu"  # "cpu", "cuda", or "mps"
    num_workers: int = 4
    use_quantization: bool = False  # For Raspberry Pi
    
    # Paths
    models_dir: str = "./models"
    cache_dir: str = "./cache"
    logs_dir: str = "./logs"
    
    # Logging
    log_level: str = "INFO"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_size: int = 500 * 1024 * 1024  # 500MB

@dataclass
class EchoNotesConfig:
    """Main configuration class combining all settings"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def for_raspberry_pi(cls) -> 'EchoNotesConfig':
        """Optimized configuration for Raspberry Pi deployment"""
        config = cls()
        config.asr.use_small_model = True
        config.system.use_quantization = True
        config.system.num_workers = 2
        config.audio.chunk_duration = 15.0  # Smaller chunks for limited RAM
        config.nlp.summarizer_model = config.nlp.small_summarizer_model
        return config
    
    @classmethod
    def for_gpu(cls) -> 'EchoNotesConfig':
        """Configuration for GPU-enabled systems"""
        config = cls()
        config.system.device = "cuda"
        config.system.num_workers = 8
        config.audio.chunk_duration = 60.0
        return config

# Global config instance
_config: Optional[EchoNotesConfig] = None

def get_config() -> EchoNotesConfig:
    """Get or create the global configuration"""
    global _config
    if _config is None:
        _config = EchoNotesConfig()
    return _config

def set_config(config: EchoNotesConfig) -> None:
    """Set the global configuration"""
    global _config
    _config = config
