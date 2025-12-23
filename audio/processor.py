"""
AudioProcessor - Signal processing and Voice Activity Detection
Handles: Filtering, normalization, noise reduction, VAD segmentation
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .capture import AudioData


@dataclass
class SpeechSegment:
    """Represents a detected speech segment"""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    samples: np.ndarray
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class AudioProcessor:
    """
    Audio signal processor with filtering, normalization, and VAD
    
    Features:
    - High-pass/low-pass/band-pass filtering
    - Audio normalization
    - Basic noise reduction
    - Voice Activity Detection (energy-based + zero-crossing)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 2
    ):
        """
        Initialize AudioProcessor
        
        Args:
            sample_rate: Audio sample rate
            frame_duration_ms: Frame duration for VAD (10, 20, or 30 ms)
            vad_aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.vad_aggressiveness = vad_aggressiveness
        
        # Energy threshold multipliers based on aggressiveness
        self._energy_multipliers = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0}
    
    def normalize(
        self,
        audio_data: AudioData,
        target_db: float = -20.0,
        headroom_db: float = 1.0
    ) -> AudioData:
        """
        Normalize audio to target dB level
        
        Args:
            audio_data: Input audio
            target_db: Target loudness in dB
            headroom_db: Headroom to prevent clipping
            
        Returns:
            Normalized AudioData
        """
        samples = audio_data.samples.copy()
        
        # Calculate current RMS
        rms = np.sqrt(np.mean(samples ** 2))
        if rms == 0:
            return audio_data
        
        current_db = 20 * np.log10(rms)
        
        # Calculate gain
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        # Apply gain with headroom
        samples = samples * gain
        max_val = np.max(np.abs(samples))
        max_allowed = 10 ** (-headroom_db / 20)
        
        if max_val > max_allowed:
            samples = samples * (max_allowed / max_val)
        
        return AudioData(
            samples=samples.astype(np.float32),
            sample_rate=audio_data.sample_rate,
            channels=audio_data.channels,
            duration=audio_data.duration,
            source=audio_data.source,
            filepath=audio_data.filepath
        )
    
    def apply_filter(
        self,
        audio_data: AudioData,
        filter_type: str = 'highpass',
        cutoff_freq: float = 80.0,
        order: int = 5
    ) -> AudioData:
        """
        Apply frequency filter to audio
        
        Args:
            audio_data: Input audio
            filter_type: 'highpass', 'lowpass', or 'bandpass'
            cutoff_freq: Cutoff frequency (or tuple for bandpass)
            order: Filter order
            
        Returns:
            Filtered AudioData
        """
        try:
            from scipy.signal import butter, filtfilt
        except ImportError:
            # Return unfiltered if scipy not available
            return audio_data
        
        nyquist = audio_data.sample_rate / 2
        
        if filter_type == 'bandpass':
            if isinstance(cutoff_freq, tuple):
                low, high = cutoff_freq
            else:
                low, high = 80.0, 8000.0
            normalized = (low / nyquist, high / nyquist)
        else:
            normalized = cutoff_freq / nyquist
        
        # Ensure normalized frequency is valid
        if isinstance(normalized, tuple):
            normalized = (max(0.001, min(normalized[0], 0.999)),
                         max(0.001, min(normalized[1], 0.999)))
        else:
            normalized = max(0.001, min(normalized, 0.999))
        
        b, a = butter(order, normalized, btype=filter_type)
        filtered = filtfilt(b, a, audio_data.samples)
        
        return AudioData(
            samples=filtered.astype(np.float32),
            sample_rate=audio_data.sample_rate,
            channels=audio_data.channels,
            duration=audio_data.duration,
            source=audio_data.source,
            filepath=audio_data.filepath
        )
    
    def reduce_noise_simple(
        self,
        audio_data: AudioData,
        noise_clip_duration: float = 0.5,
        strength: float = 0.7
    ) -> AudioData:
        """
        Simple spectral subtraction noise reduction
        
        Uses the first portion of audio as noise profile.
        For better results, use IntelligentAudioPreprocessor.
        
        Args:
            audio_data: Input audio
            noise_clip_duration: Duration of noise sample at start
            strength: Noise reduction strength (0-1)
            
        Returns:
            Noise-reduced AudioData
        """
        try:
            from scipy.fft import rfft, irfft
        except ImportError:
            return audio_data
        
        samples = audio_data.samples.copy()
        noise_samples = int(noise_clip_duration * audio_data.sample_rate)
        
        if len(samples) <= noise_samples:
            return audio_data
        
        # Estimate noise profile from beginning
        noise_profile = samples[:noise_samples]
        noise_spectrum = np.abs(rfft(noise_profile))
        
        # Process in frames
        frame_size = 1024
        hop_size = frame_size // 2
        output = np.zeros_like(samples)
        window = np.hanning(frame_size)
        
        for i in range(0, len(samples) - frame_size, hop_size):
            frame = samples[i:i + frame_size] * window
            spectrum = rfft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # Pad noise spectrum if needed
            if len(noise_spectrum) < len(magnitude):
                noise_pad = np.zeros(len(magnitude))
                noise_pad[:len(noise_spectrum)] = noise_spectrum
                noise_spectrum = noise_pad
            
            # Spectral subtraction
            clean_magnitude = magnitude - strength * noise_spectrum[:len(magnitude)]
            clean_magnitude = np.maximum(clean_magnitude, 0.01 * magnitude)  # Flooring
            
            # Reconstruct
            clean_spectrum = clean_magnitude * np.exp(1j * phase)
            clean_frame = irfft(clean_spectrum, n=frame_size)
            
            output[i:i + frame_size] += clean_frame * window
        
        # Normalize overlap-add
        output = output / (frame_size / hop_size / 2)
        
        return AudioData(
            samples=output.astype(np.float32),
            sample_rate=audio_data.sample_rate,
            channels=audio_data.channels,
            duration=audio_data.duration,
            source=audio_data.source,
            filepath=audio_data.filepath
        )
    
    def detect_voice_activity(
        self,
        audio_data: AudioData,
        min_speech_duration: float = 0.3,
        max_silence_duration: float = 0.5
    ) -> List[SpeechSegment]:
        """
        Detect speech segments using energy and zero-crossing rate
        
        Args:
            audio_data: Input audio
            min_speech_duration: Minimum speech segment duration
            max_silence_duration: Maximum silence within speech
            
        Returns:
            List of SpeechSegment objects
        """
        samples = audio_data.samples
        sr = audio_data.sample_rate
        
        # Frame-by-frame analysis
        frame_energies = []
        frame_zcrs = []
        
        for i in range(0, len(samples) - self.frame_size, self.frame_size):
            frame = samples[i:i + self.frame_size]
            
            # Short-time energy
            energy = np.sum(frame ** 2) / self.frame_size
            frame_energies.append(energy)
            
            # Zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * self.frame_size)
            frame_zcrs.append(zcr)
        
        frame_energies = np.array(frame_energies)
        frame_zcrs = np.array(frame_zcrs)
        
        # Adaptive thresholds
        energy_threshold = np.mean(frame_energies) * self._energy_multipliers[self.vad_aggressiveness]
        zcr_threshold = np.mean(frame_zcrs) * 1.5
        
        # Classify frames
        is_speech = (frame_energies > energy_threshold) | (frame_zcrs < zcr_threshold)
        
        # Smooth with median filter
        kernel_size = max(3, int(0.1 * sr / self.frame_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        try:
            from scipy.ndimage import median_filter
            is_speech = median_filter(is_speech.astype(float), size=kernel_size) > 0.5
        except ImportError:
            pass
        
        # Convert to segments
        segments = []
        in_speech = False
        start_frame = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                end_frame = i
                start_time = start_frame * self.frame_size / sr
                end_time = end_frame * self.frame_size / sr
                
                if end_time - start_time >= min_speech_duration:
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segments.append(SpeechSegment(
                        start_time=start_time,
                        end_time=end_time,
                        samples=samples[start_sample:end_sample]
                    ))
                in_speech = False
        
        # Handle last segment
        if in_speech:
            end_time = len(samples) / sr
            start_time = start_frame * self.frame_size / sr
            if end_time - start_time >= min_speech_duration:
                start_sample = int(start_time * sr)
                segments.append(SpeechSegment(
                    start_time=start_time,
                    end_time=end_time,
                    samples=samples[start_sample:]
                ))
        
        # Merge close segments
        merged_segments = self._merge_segments(segments, max_silence_duration, sr)
        
        return merged_segments
    
    def _merge_segments(
        self,
        segments: List[SpeechSegment],
        max_gap: float,
        sample_rate: int
    ) -> List[SpeechSegment]:
        """Merge segments that are close together"""
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.start_time - prev.end_time
            
            if gap <= max_gap:
                # Merge segments
                merged[-1] = SpeechSegment(
                    start_time=prev.start_time,
                    end_time=seg.end_time,
                    samples=np.concatenate([prev.samples, seg.samples])
                )
            else:
                merged.append(seg)
        
        return merged
    
    def compute_snr(self, audio_data: AudioData, noise_duration: float = 0.5) -> float:
        """
        Estimate Signal-to-Noise Ratio
        
        Args:
            audio_data: Input audio
            noise_duration: Duration at start to use as noise estimate
            
        Returns:
            Estimated SNR in dB
        """
        samples = audio_data.samples
        noise_samples = int(noise_duration * audio_data.sample_rate)
        
        if len(samples) <= noise_samples * 2:
            return 0.0
        
        noise = samples[:noise_samples]
        signal = samples[noise_samples:]
        
        noise_power = np.mean(noise ** 2)
        signal_power = np.mean(signal ** 2)
        
        if noise_power == 0:
            return 100.0  # Very clean
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    
    def split_into_chunks(
        self,
        audio_data: AudioData,
        chunk_duration: float = 30.0,
        overlap: float = 1.0
    ) -> List[AudioData]:
        """
        Split audio into overlapping chunks for processing
        
        Args:
            audio_data: Input audio
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of AudioData chunks
        """
        samples = audio_data.samples
        sr = audio_data.sample_rate
        
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap * sr)
        step = chunk_samples - overlap_samples
        
        chunks = []
        for i in range(0, len(samples), step):
            end = min(i + chunk_samples, len(samples))
            chunk_data = samples[i:end]
            
            if len(chunk_data) < sr * 0.5:  # Skip very short final chunks
                break
            
            chunks.append(AudioData(
                samples=chunk_data,
                sample_rate=sr,
                channels=audio_data.channels,
                duration=len(chunk_data) / sr,
                source=audio_data.source,
                filepath=audio_data.filepath
            ))
        
        return chunks
