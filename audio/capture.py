"""
AudioCapture - Multi-source audio acquisition
Supports: File upload, microphone recording, streaming
"""
import io
import wave
import tempfile
from pathlib import Path
from typing import Optional, Union, Tuple, Generator, BinaryIO
from dataclasses import dataclass
import struct

import numpy as np

# Lazy imports for optional dependencies
def _get_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except ImportError:
        raise ImportError("sounddevice required for microphone capture. Install: pip install sounddevice")

def _get_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        raise ImportError("soundfile required for audio file reading. Install: pip install soundfile")


@dataclass
class AudioData:
    """Container for audio data with metadata"""
    samples: np.ndarray  # Audio samples as numpy array
    sample_rate: int
    channels: int
    duration: float  # Duration in seconds
    source: str  # 'file', 'microphone', 'stream'
    filepath: Optional[str] = None
    
    @property
    def is_mono(self) -> bool:
        return self.channels == 1
    
    @property
    def num_samples(self) -> int:
        return len(self.samples)
    
    def to_mono(self) -> 'AudioData':
        """Convert stereo to mono by averaging channels"""
        if self.is_mono:
            return self
        if len(self.samples.shape) > 1:
            mono_samples = np.mean(self.samples, axis=1)
        else:
            mono_samples = self.samples
        return AudioData(
            samples=mono_samples,
            sample_rate=self.sample_rate,
            channels=1,
            duration=self.duration,
            source=self.source,
            filepath=self.filepath
        )


class AudioCapture:
    """
    Multi-source audio capture handler
    
    Supports:
    - File loading (WAV, MP3, FLAC, OGG, M4A)
    - Microphone recording
    - Audio streaming
    """
    
    SUPPORTED_FORMATS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm', 'wma', 'aac'}
    
    def __init__(self, target_sample_rate: int = 16000, target_channels: int = 1):
        """
        Initialize AudioCapture
        
        Args:
            target_sample_rate: Target sample rate for output (default 16kHz for ASR)
            target_channels: Target number of channels (default 1 for mono)
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
    
    def load_file(self, filepath: Union[str, Path]) -> AudioData:
        """
        Load audio from a file
        
        Args:
            filepath: Path to audio file
            
        Returns:
            AudioData object containing the loaded audio
        """
        sf = _get_soundfile()
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        suffix = filepath.suffix.lower().lstrip('.')
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {suffix}. Supported: {self.SUPPORTED_FORMATS}")
        
        # Load audio file
        samples, sample_rate = sf.read(str(filepath), dtype='float32')
        
        # Determine number of channels
        if len(samples.shape) == 1:
            channels = 1
        else:
            channels = samples.shape[1]
        
        duration = len(samples) / sample_rate
        
        audio_data = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            channels=channels,
            duration=duration,
            source='file',
            filepath=str(filepath)
        )
        
        # Convert to target format
        return self._convert_to_target(audio_data)
    
    def load_bytes(self, audio_bytes: bytes, format_hint: str = 'wav') -> AudioData:
        """
        Load audio from bytes (useful for web uploads)
        
        Args:
            audio_bytes: Raw audio bytes
            format_hint: Format of the audio data
            
        Returns:
            AudioData object
        """
        sf = _get_soundfile()
        
        # Create a temporary file to handle the bytes
        with tempfile.NamedTemporaryFile(suffix=f'.{format_hint}', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            audio_data = self.load_file(tmp_path)
            audio_data.source = 'bytes'
            return audio_data
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def record_microphone(
        self,
        duration: float,
        device: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> AudioData:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            device: Audio device index (None for default)
            callback: Optional callback for progress updates
            
        Returns:
            AudioData object with recorded audio
        """
        sd = _get_sounddevice()
        
        num_samples = int(duration * self.target_sample_rate)
        
        # Record audio
        recording = sd.rec(
            num_samples,
            samplerate=self.target_sample_rate,
            channels=self.target_channels,
            dtype='float32',
            device=device
        )
        sd.wait()  # Wait until recording is finished
        
        # Flatten if mono
        if self.target_channels == 1:
            recording = recording.flatten()
        
        return AudioData(
            samples=recording,
            sample_rate=self.target_sample_rate,
            channels=self.target_channels,
            duration=duration,
            source='microphone'
        )
    
    def stream_microphone(
        self,
        chunk_duration: float = 0.5,
        device: Optional[int] = None
    ) -> Generator[AudioData, None, None]:
        """
        Stream audio from microphone in chunks
        
        Args:
            chunk_duration: Duration of each chunk in seconds
            device: Audio device index
            
        Yields:
            AudioData objects for each chunk
        """
        sd = _get_sounddevice()
        
        chunk_samples = int(chunk_duration * self.target_sample_rate)
        
        with sd.InputStream(
            samplerate=self.target_sample_rate,
            channels=self.target_channels,
            dtype='float32',
            device=device,
            blocksize=chunk_samples
        ) as stream:
            while True:
                chunk, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("Warning: Audio buffer overflow")
                
                if self.target_channels == 1:
                    chunk = chunk.flatten()
                
                yield AudioData(
                    samples=chunk,
                    sample_rate=self.target_sample_rate,
                    channels=self.target_channels,
                    duration=chunk_duration,
                    source='stream'
                )
    
    def _convert_to_target(self, audio_data: AudioData) -> AudioData:
        """Convert audio to target sample rate and channels"""
        samples = audio_data.samples
        
        # Convert to mono if needed
        if audio_data.channels > 1 and self.target_channels == 1:
            if len(samples.shape) > 1:
                samples = np.mean(samples, axis=1)
        
        # Resample if needed
        if audio_data.sample_rate != self.target_sample_rate:
            samples = self._resample(samples, audio_data.sample_rate, self.target_sample_rate)
        
        duration = len(samples) / self.target_sample_rate
        
        return AudioData(
            samples=samples.astype(np.float32),
            sample_rate=self.target_sample_rate,
            channels=self.target_channels,
            duration=duration,
            source=audio_data.source,
            filepath=audio_data.filepath
        )
    
    def _resample(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using scipy"""
        try:
            from scipy import signal
            num_samples = int(len(samples) * target_sr / orig_sr)
            resampled = signal.resample(samples, num_samples)
            return resampled
        except ImportError:
            # Fallback: simple linear interpolation
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(samples), 1/ratio)
            indices = indices[indices < len(samples) - 1].astype(int)
            return samples[indices]
    
    @staticmethod
    def list_devices() -> list:
        """List available audio input devices"""
        sd = _get_sounddevice()
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate']
                })
        return input_devices
    
    def save_audio(self, audio_data: AudioData, filepath: Union[str, Path]) -> str:
        """
        Save audio data to file
        
        Args:
            audio_data: AudioData object to save
            filepath: Output file path
            
        Returns:
            Path to saved file
        """
        sf = _get_soundfile()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(
            str(filepath),
            audio_data.samples,
            audio_data.sample_rate,
            format=filepath.suffix.lstrip('.').upper()
        )
        
        return str(filepath)
