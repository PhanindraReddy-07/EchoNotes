"""
Audio Processor Module
Handles audio preprocessing: resampling, normalization, noise reduction.
"""

import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import tempfile


class AudioProcessor:
    """Processes audio for optimal speech recognition."""
    
    def __init__(self, settings):
        self.settings = settings
        audio_config = settings.config.get('audio', {})
        self.target_sample_rate = audio_config.get('sample_rate', 16000) or 16000
    
    def process(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Full audio processing pipeline.
        
        Args:
            audio_data: Raw audio as numpy array
            sample_rate: Original sample rate
            
        Returns:
            Processed audio data
        """
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            audio_data = self.resample(audio_data, sample_rate, self.target_sample_rate)
        
        # Apply high-pass filter to remove low frequency noise
        audio_data = self.highpass_filter(audio_data, cutoff=80)
        
        # Normalize
        audio_data = self.normalize(audio_data)
        
        # Apply light noise reduction
        audio_data = self.reduce_noise(audio_data)
        
        return audio_data
    
    def resample(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio_data
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        new_length = int(len(audio_data) * ratio)
        
        # Use scipy's resample
        resampled = signal.resample(audio_data, new_length)
        
        return resampled.astype(np.float32)
    
    def highpass_filter(self, audio_data: np.ndarray, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter to remove low frequency noise (hum, rumble)."""
        nyquist = self.target_sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1.0:
            return audio_data
        
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        filtered = signal.filtfilt(b, a, audio_data)
        
        return filtered.astype(np.float32)
    
    def normalize(self, audio_data: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target dB level (peak normalization)."""
        # Find peak
        peak = np.max(np.abs(audio_data))
        
        if peak > 0:
            # Calculate target peak
            target_peak = 10 ** (target_db / 20)
            
            # Scale audio
            audio_data = audio_data * (target_peak / peak)
            
            # Clip to prevent clipping
            audio_data = np.clip(audio_data, -1.0, 1.0)
        
        return audio_data.astype(np.float32)
    
    def reduce_noise(self, audio_data: np.ndarray, 
                     noise_threshold: float = 0.01) -> np.ndarray:
        """
        Simple noise reduction using spectral gating.
        
        Args:
            audio_data: Input audio
            noise_threshold: Threshold for noise gate
            
        Returns:
            Noise-reduced audio
        """
        # Simple noise gate - reduce very quiet parts
        mask = np.abs(audio_data) > noise_threshold
        audio_data = audio_data * mask + audio_data * ~mask * 0.05
        
        return audio_data.astype(np.float32)
    
    def apply_vad(self, audio_data: np.ndarray, 
                  frame_duration: float = 0.03,
                  energy_threshold: float = 0.01) -> np.ndarray:
        """
        Voice Activity Detection - removes silence.
        
        Args:
            audio_data: Input audio
            frame_duration: Frame size in seconds
            energy_threshold: Energy threshold for speech detection
            
        Returns:
            Audio with silence removed
        """
        frame_size = int(self.target_sample_rate * frame_duration)
        num_frames = len(audio_data) // frame_size
        
        if num_frames == 0:
            return audio_data
        
        # Calculate energy per frame
        frames = audio_data[:num_frames * frame_size].reshape(-1, frame_size)
        energies = np.mean(frames ** 2, axis=1)
        
        # Find speech frames
        speech_frames = energies > energy_threshold
        
        # Expand speech regions slightly
        expanded = np.copy(speech_frames)
        for i in range(1, len(speech_frames) - 1):
            if speech_frames[i-1] or speech_frames[i+1]:
                expanded[i] = True
        
        # Extract speech segments
        speech_audio = frames[expanded].flatten()
        
        return speech_audio.astype(np.float32) if len(speech_audio) > 0 else audio_data
    
    def to_int16(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert float32 audio to int16 for Vosk."""
        # Scale to int16 range
        audio_int16 = (audio_data * 32767).astype(np.int16)
        return audio_int16
    
    def save_temp_wav(self, audio_data: np.ndarray) -> Path:
        """Save audio to temporary WAV file (for debugging)."""
        temp_dir = Path(tempfile.gettempdir())
        temp_path = temp_dir / 'echonotes_temp.wav'
        
        audio_int16 = self.to_int16(audio_data)
        wavfile.write(temp_path, self.target_sample_rate, audio_int16)
        
        return temp_path