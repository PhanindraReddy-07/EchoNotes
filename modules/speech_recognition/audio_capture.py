"""
Audio Capture Module
Uses sounddevice for cross-platform audio capture (Windows, Mac, Linux).
"""

import queue
from pathlib import Path
from typing import Generator, Tuple
import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioCapture:
    """Handles audio input from files and microphone."""
    
    def __init__(self, settings):
        self.settings = settings
        
        # Get audio settings with fallback defaults
        audio_config = settings.config.get('audio', {})
        self.sample_rate = audio_config.get('sample_rate', 16000) or 16000
        self.channels = audio_config.get('channels', 1) or 1
        self.chunk_size = audio_config.get('chunk_size', 4096) or 4096
        
        # Audio queue for streaming
        self._audio_queue = queue.Queue()
    
    def load_file(self, path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio from file.
        
        Supports: WAV, FLAC, OGG, MP3 (via soundfile)
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        # Load audio file
        audio_data, file_sample_rate = sf.read(path, dtype='float32')
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return audio_data, file_sample_rate
    
    def record(self, duration: int) -> Tuple[np.ndarray, int]:
        """
        Record audio from default microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        print(f"Recording for {duration} seconds...")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        
        # Wait for recording to complete
        sd.wait()
        
        # Flatten if mono
        if self.channels == 1:
            audio_data = audio_data.flatten()
        
        return audio_data, self.sample_rate
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """
        Stream audio from microphone in real-time.
        
        Yields:
            Audio chunks as numpy arrays
        """
        def audio_callback(indata, frames, time, status):
            """Callback for audio stream."""
            if status:
                print(f"Audio status: {status}")
            self._audio_queue.put(indata.copy())
        
        # Clear queue
        while not self._audio_queue.empty():
            self._audio_queue.get()
        
        # Start stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=audio_callback
        ):
            while True:
                try:
                    chunk = self._audio_queue.get(timeout=1.0)
                    yield chunk.flatten() if self.channels == 1 else chunk
                except queue.Empty:
                    continue
    
    def list_devices(self):
        """List available audio input devices."""
        print("\nAvailable Audio Devices:")
        print("-" * 50)
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']}")
                print(f"      Input channels: {device['max_input_channels']}")
                print(f"      Sample rate: {device['default_samplerate']}")
        print("-" * 50)
        return devices
    
    def set_device(self, device_id: int):
        """Set the input device by ID."""
        sd.default.device[0] = device_id
        print(f"Input device set to: {sd.query_devices(device_id)['name']}")