"""
EchoNotes Configuration Settings
Handles YAML config loading with Windows path support.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Settings:
    """Configuration manager for EchoNotes."""
    
    DEFAULT_CONFIG = {
        'audio': {
            'sample_rate': 16000,
            'channels': 1,
            'chunk_size': 4096,
            'format': 'int16',
        },
        'speech': {
            'model_size': 'small',  # small, medium, large
            'language': 'en-us',
        },
        'nlp': {
            'use_gpu': False,
            'summarizer_model': 'sshleifer/distilbart-cnn-12-6',
            'summarizer_model_small': 'google/flan-t5-small',
            'max_summary_length': 150,
            'min_summary_length': 50,
            'use_extractive_fallback': True,
        },
        'document': {
            'default_format': 'markdown',
            'include_entities': True,
            'include_summary': True,
            'include_timestamps': True,
        },
        'paths': {
            'models_dir': 'models',
            'output_dir': 'output',
            'temp_dir': 'temp',
        },
        'performance': {
            'low_memory_mode': False,
            'batch_size': 4,
            'num_threads': 4,
        },
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load config file if exists
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            default_config = self.project_root / 'config' / 'config.yaml'
            if default_config.exists():
                self._load_config(default_config)
        
        # Resolve paths
        self._resolve_paths()
        
        # Set environment variables for better Windows compatibility
        self._setup_environment()
    
    def _load_config(self, path: Path):
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        
        if user_config:
            self._deep_update(self.config, user_config)
    
    def _deep_update(self, base: dict, update: dict):
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _resolve_paths(self):
        """Convert relative paths to absolute Windows-compatible paths."""
        paths = self.config['paths']
        for key in paths:
            path = Path(paths[key])
            if not path.is_absolute():
                paths[key] = str(self.project_root / path)
            
            # Create directory if it's a dir path
            if key.endswith('_dir'):
                Path(paths[key]).mkdir(parents=True, exist_ok=True)
    
    def _setup_environment(self):
        """Set environment variables for Windows compatibility."""
        # Limit threads for CPU-bound tasks
        threads = str(self.config['performance']['num_threads'])
        os.environ.setdefault('OMP_NUM_THREADS', threads)
        os.environ.setdefault('MKL_NUM_THREADS', threads)
        
        # Disable tokenizers parallelism warning
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    def get(self, *keys, default=None) -> Any:
        """Get nested config value using dot notation or multiple keys."""
        if len(keys) == 1 and '.' in keys[0]:
            keys = keys[0].split('.')
        
        value = self.config
        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value if value is not None else default
        except (KeyError, TypeError):
            return default
    
    @property
    def models_dir(self) -> Path:
        return Path(self.config['paths']['models_dir'])
    
    @property
    def output_dir(self) -> Path:
        return Path(self.config['paths']['output_dir'])
    
    @property
    def vosk_model_path(self) -> Path:
        """Get path to Vosk model."""
        model_size = self.config['speech']['model_size']
        return self.models_dir / f'vosk-model-{model_size}'
    
    @property
    def use_gpu(self) -> bool:
        return self.config['nlp']['use_gpu']
    
    @property
    def low_memory_mode(self) -> bool:
        return self.config['performance']['low_memory_mode']