"""
Transcriber Module
Offline speech-to-text using Vosk.
"""

import json
from pathlib import Path
from typing import Generator
import numpy as np

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("Warning: Vosk not installed. Run: pip install vosk")


class Transcriber:
    """Offline speech-to-text using Vosk."""
    
    def __init__(self, settings):
        self.settings = settings
        audio_config = settings.config.get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 16000) or 16000
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Vosk model."""
        if not VOSK_AVAILABLE:
            return
        
        model_path = self.settings.vosk_model_path
        
        if not model_path.exists():
            print(f"Warning: Vosk model not found at {model_path}")
            print("Run: python setup_models.py")
            return
        
        try:
            self.model = Model(str(model_path))
            print(f"Loaded Vosk model: {model_path.name}")
        except Exception as e:
            print(f"Error loading Vosk model: {e}")
    
    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Processed audio as numpy array (float32)
            
        Returns:
            Transcribed text
        """
        if self.model is None:
            return "[Error: Vosk model not loaded]"
        
        # Convert to int16 bytes for Vosk
        if audio_data.dtype == np.float32:
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_data.astype(np.int16).tobytes()
        
        # Create recognizer
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)
        
        # Process audio in chunks
        chunk_size = self.sample_rate * 2  # 1 second chunks (16-bit = 2 bytes)
        results = []
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                if result.get('text'):
                    results.append(result['text'])
        
        # Get final result
        final_result = json.loads(recognizer.FinalResult())
        if final_result.get('text'):
            results.append(final_result['text'])
        
        return ' '.join(results)
    
    def transcribe_with_timestamps(self, audio_data: np.ndarray) -> list:
        """
        Transcribe audio with word-level timestamps.
        
        Returns:
            List of dicts with 'word', 'start', 'end' keys
        """
        if self.model is None:
            return []
        
        # Convert to int16 bytes
        if audio_data.dtype == np.float32:
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        else:
            audio_bytes = audio_data.astype(np.int16).tobytes()
        
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)
        
        words_with_times = []
        chunk_size = self.sample_rate * 2
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                if 'result' in result:
                    words_with_times.extend(result['result'])
        
        final_result = json.loads(recognizer.FinalResult())
        if 'result' in final_result:
            words_with_times.extend(final_result['result'])
        
        return words_with_times
    
    def stream_transcribe(self, audio_stream: Generator) -> Generator[str, None, None]:
        """
        Transcribe streaming audio in real-time.
        
        Args:
            audio_stream: Generator yielding audio chunks
            
        Yields:
            Transcribed text chunks
        """
        if self.model is None:
            yield "[Error: Vosk model not loaded]"
            return
        
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)
        
        for audio_chunk in audio_stream:
            # Convert to int16 bytes
            if audio_chunk.dtype == np.float32:
                chunk_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            else:
                chunk_bytes = audio_chunk.astype(np.int16).tobytes()
            
            if recognizer.AcceptWaveform(chunk_bytes):
                result = json.loads(recognizer.Result())
                if result.get('text'):
                    yield result['text']
            else:
                # Partial result (optional)
                partial = json.loads(recognizer.PartialResult())
                if partial.get('partial'):
                    # Could yield partial for real-time display
                    pass
    
    @property
    def is_available(self) -> bool:
        """Check if transcriber is ready."""
        return VOSK_AVAILABLE and self.model is not None