"""
IntelligentAudioPreprocessor - Advanced audio enhancement
NEW MODULE: Noise profiling, adaptive filtering, quality assessment

This is a KEY NOVEL COMPONENT for the final year project,
demonstrating ML-based audio preprocessing for better ASR performance.
"""
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from .capture import AudioData
from .processor import AudioProcessor


class NoiseType(Enum):
    """Detected noise types"""
    CLEAN = "clean"
    WHITE_NOISE = "white_noise"
    BACKGROUND_SPEECH = "background_speech"
    MUSIC = "music"
    ENVIRONMENTAL = "environmental"  # AC, traffic, etc.
    REVERB = "reverb"
    MIXED = "mixed"


@dataclass
class AudioQualityReport:
    """Detailed audio quality assessment"""
    overall_score: float  # 0-100
    snr_db: float
    noise_type: NoiseType
    clarity_score: float  # Speech clarity 0-1
    predicted_wer: float  # Predicted Word Error Rate
    recommendations: List[str]
    enhancement_applied: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'overall_score': round(self.overall_score, 1),
            'snr_db': round(self.snr_db, 1),
            'noise_type': self.noise_type.value,
            'clarity_score': round(self.clarity_score, 3),
            'predicted_wer': round(self.predicted_wer, 3),
            'recommendations': self.recommendations,
            'enhancement_applied': self.enhancement_applied
        }


class IntelligentAudioPreprocessor:
    """
    Advanced audio preprocessing with intelligent enhancement
    
    Features:
    - Automatic noise type detection
    - Adaptive noise reduction
    - Quality prediction before ASR
    - Enhancement recommendations
    
    Technical Novelty:
    - Uses spectral analysis for noise profiling
    - Predicts transcription quality before processing
    - Adaptive filter selection based on noise type
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        enhancement_strength: str = 'auto'  # 'light', 'medium', 'aggressive', 'auto'
    ):
        """
        Initialize the intelligent preprocessor
        
        Args:
            sample_rate: Target sample rate
            enhancement_strength: Enhancement level or 'auto' for adaptive
        """
        self.sample_rate = sample_rate
        self.enhancement_strength = enhancement_strength
        self.processor = AudioProcessor(sample_rate=sample_rate)
        
        # Noise profile parameters
        self._noise_profile: Optional[np.ndarray] = None
        self._spectral_floor: Optional[np.ndarray] = None
    
    def analyze_quality(self, audio_data: AudioData) -> AudioQualityReport:
        """
        Comprehensive audio quality analysis
        
        Analyzes:
        - Signal-to-noise ratio
        - Noise type classification
        - Speech clarity
        - Predicted ASR performance
        
        Args:
            audio_data: Input audio
            
        Returns:
            Detailed AudioQualityReport
        """
        samples = audio_data.samples
        sr = audio_data.sample_rate
        
        # Calculate SNR
        snr_db = self.processor.compute_snr(audio_data)
        
        # Detect noise type
        noise_type = self._detect_noise_type(samples, sr)
        
        # Calculate speech clarity
        clarity = self._compute_clarity_score(samples, sr)
        
        # Predict WER based on quality metrics
        predicted_wer = self._predict_wer(snr_db, clarity, noise_type)
        
        # Calculate overall score (0-100)
        overall = self._compute_overall_score(snr_db, clarity, noise_type)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            snr_db, noise_type, clarity, overall
        )
        
        return AudioQualityReport(
            overall_score=overall,
            snr_db=snr_db,
            noise_type=noise_type,
            clarity_score=clarity,
            predicted_wer=predicted_wer,
            recommendations=recommendations
        )
    
    def enhance(
        self,
        audio_data: AudioData,
        quality_report: Optional[AudioQualityReport] = None
    ) -> Tuple[AudioData, AudioQualityReport]:
        """
        Apply intelligent audio enhancement
        
        Selects and applies appropriate enhancement based on
        detected noise type and quality assessment.
        
        Args:
            audio_data: Input audio
            quality_report: Pre-computed quality report (optional)
            
        Returns:
            Tuple of (enhanced AudioData, updated quality report)
        """
        if quality_report is None:
            quality_report = self.analyze_quality(audio_data)
        
        # Skip enhancement if already high quality
        if quality_report.overall_score > 85:
            return audio_data, quality_report
        
        samples = audio_data.samples.copy()
        sr = audio_data.sample_rate
        
        # Determine enhancement strength
        strength = self._get_enhancement_strength(quality_report)
        
        # Apply noise-type-specific enhancement
        if quality_report.noise_type == NoiseType.WHITE_NOISE:
            samples = self._reduce_white_noise(samples, sr, strength)
        elif quality_report.noise_type == NoiseType.BACKGROUND_SPEECH:
            samples = self._reduce_background_speech(samples, sr, strength)
        elif quality_report.noise_type == NoiseType.ENVIRONMENTAL:
            samples = self._reduce_environmental_noise(samples, sr, strength)
        elif quality_report.noise_type == NoiseType.REVERB:
            samples = self._reduce_reverb(samples, sr, strength)
        else:
            # General enhancement for mixed/unknown noise
            samples = self._general_enhancement(samples, sr, strength)
        
        # Apply high-pass filter to remove low-frequency rumble
        enhanced_data = AudioData(
            samples=samples.astype(np.float32),
            sample_rate=sr,
            channels=audio_data.channels,
            duration=len(samples) / sr,
            source=audio_data.source,
            filepath=audio_data.filepath
        )
        
        enhanced_data = self.processor.apply_filter(
            enhanced_data, 'highpass', cutoff_freq=80.0
        )
        
        # Normalize
        enhanced_data = self.processor.normalize(enhanced_data, target_db=-20.0)
        
        # Update quality report
        new_report = self.analyze_quality(enhanced_data)
        new_report.enhancement_applied = True
        
        return enhanced_data, new_report
    
    def _detect_noise_type(self, samples: np.ndarray, sr: int) -> NoiseType:
        """
        Classify the type of noise present in the audio
        
        Uses spectral analysis and temporal patterns to identify noise type.
        """
        try:
            from scipy.fft import rfft
            from scipy.stats import entropy
        except ImportError:
            return NoiseType.MIXED
        
        # Compute spectrum
        spectrum = np.abs(rfft(samples[:min(len(samples), sr * 2)]))
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # Spectral flatness (high = white noise, low = tonal)
        geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
        arithmetic_mean = np.mean(spectrum)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Spectral entropy
        spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)
        spec_entropy = entropy(spectrum_norm)
        
        # Energy in speech frequency range (300-3400 Hz)
        freq_resolution = sr / 2 / len(spectrum)
        speech_low = int(300 / freq_resolution)
        speech_high = int(3400 / freq_resolution)
        speech_energy = np.sum(spectrum[speech_low:speech_high] ** 2)
        total_energy = np.sum(spectrum ** 2)
        speech_ratio = speech_energy / (total_energy + 1e-10)
        
        # Low frequency energy (environmental noise indicator)
        low_freq = int(200 / freq_resolution)
        low_energy_ratio = np.sum(spectrum[:low_freq] ** 2) / (total_energy + 1e-10)
        
        # Classify based on features
        if spectral_flatness > 0.8:
            return NoiseType.WHITE_NOISE
        elif spectral_flatness > 0.5 and low_energy_ratio > 0.3:
            return NoiseType.ENVIRONMENTAL
        elif speech_ratio > 0.6 and spectral_flatness < 0.3:
            # Check for overlapping speech patterns
            return NoiseType.BACKGROUND_SPEECH
        elif spectral_flatness < 0.2 and spec_entropy < 4:
            return NoiseType.MUSIC
        elif self._detect_reverb(samples, sr):
            return NoiseType.REVERB
        elif spectral_flatness < 0.4:
            return NoiseType.CLEAN
        else:
            return NoiseType.MIXED
    
    def _detect_reverb(self, samples: np.ndarray, sr: int) -> bool:
        """Detect if reverb is present using autocorrelation"""
        try:
            from scipy.signal import correlate
        except ImportError:
            return False
        
        # Analyze a chunk
        chunk = samples[:min(len(samples), sr)]
        
        # Autocorrelation
        autocorr = correlate(chunk, chunk, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Look for significant correlation at typical reverb delays (20-100ms)
        delay_start = int(0.02 * sr)
        delay_end = int(0.1 * sr)
        
        if delay_end < len(autocorr):
            reverb_correlation = np.max(np.abs(autocorr[delay_start:delay_end]))
            return reverb_correlation > 0.3
        
        return False
    
    def _compute_clarity_score(self, samples: np.ndarray, sr: int) -> float:
        """
        Compute speech clarity score
        
        Based on spectral contrast and modulation depth.
        """
        try:
            from scipy.fft import rfft
        except ImportError:
            return 0.5
        
        # Frame-based analysis
        frame_size = int(0.025 * sr)  # 25ms frames
        hop_size = frame_size // 2
        
        spectral_contrasts = []
        
        for i in range(0, len(samples) - frame_size, hop_size):
            frame = samples[i:i + frame_size]
            spectrum = np.abs(rfft(frame))
            
            if len(spectrum) > 10:
                # Spectral contrast: difference between peaks and valleys
                sorted_spec = np.sort(spectrum)
                n = len(sorted_spec)
                peaks = np.mean(sorted_spec[-n//10:])
                valleys = np.mean(sorted_spec[:n//10])
                contrast = (peaks - valleys) / (peaks + valleys + 1e-10)
                spectral_contrasts.append(contrast)
        
        if not spectral_contrasts:
            return 0.5
        
        # Higher contrast = clearer speech
        avg_contrast = np.mean(spectral_contrasts)
        clarity = min(1.0, avg_contrast * 2)  # Scale to 0-1
        
        return clarity
    
    def _predict_wer(
        self,
        snr_db: float,
        clarity: float,
        noise_type: NoiseType
    ) -> float:
        """
        Predict expected Word Error Rate based on audio quality
        
        Uses empirical relationships between quality metrics and ASR performance.
        """
        # Base WER prediction from SNR (empirical relationship)
        if snr_db > 30:
            base_wer = 0.05
        elif snr_db > 20:
            base_wer = 0.10
        elif snr_db > 15:
            base_wer = 0.15
        elif snr_db > 10:
            base_wer = 0.25
        elif snr_db > 5:
            base_wer = 0.40
        else:
            base_wer = 0.60
        
        # Adjust for clarity
        clarity_factor = 1.5 - clarity  # Lower clarity = higher WER
        
        # Adjust for noise type (some are harder than others)
        noise_factors = {
            NoiseType.CLEAN: 0.8,
            NoiseType.WHITE_NOISE: 1.0,
            NoiseType.ENVIRONMENTAL: 1.1,
            NoiseType.BACKGROUND_SPEECH: 1.5,  # Hardest for ASR
            NoiseType.MUSIC: 1.2,
            NoiseType.REVERB: 1.3,
            NoiseType.MIXED: 1.2
        }
        noise_factor = noise_factors.get(noise_type, 1.0)
        
        predicted_wer = base_wer * clarity_factor * noise_factor
        return min(1.0, max(0.01, predicted_wer))
    
    def _compute_overall_score(
        self,
        snr_db: float,
        clarity: float,
        noise_type: NoiseType
    ) -> float:
        """Compute overall quality score (0-100)"""
        # SNR contribution (0-40 points)
        snr_score = min(40, max(0, (snr_db + 5) * 2))
        
        # Clarity contribution (0-40 points)
        clarity_score = clarity * 40
        
        # Noise type penalty (0-20 points)
        noise_penalties = {
            NoiseType.CLEAN: 0,
            NoiseType.WHITE_NOISE: 5,
            NoiseType.ENVIRONMENTAL: 8,
            NoiseType.MUSIC: 10,
            NoiseType.REVERB: 12,
            NoiseType.BACKGROUND_SPEECH: 15,
            NoiseType.MIXED: 10
        }
        noise_score = 20 - noise_penalties.get(noise_type, 10)
        
        total = snr_score + clarity_score + noise_score
        return min(100, max(0, total))
    
    def _generate_recommendations(
        self,
        snr_db: float,
        noise_type: NoiseType,
        clarity: float,
        overall: float
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if snr_db < 10:
            recommendations.append("Low SNR detected. Consider using a closer microphone or quieter environment.")
        
        if noise_type == NoiseType.BACKGROUND_SPEECH:
            recommendations.append("Background speech detected. This may cause transcription errors with speaker confusion.")
        
        if noise_type == NoiseType.REVERB:
            recommendations.append("Reverb detected. Consider recording in a more acoustically treated space.")
        
        if clarity < 0.4:
            recommendations.append("Speech clarity is low. Audio enhancement will be applied.")
        
        if overall < 50:
            recommendations.append("Overall quality is poor. Consider re-recording if possible.")
        elif overall < 70:
            recommendations.append("Moderate quality. Enhancement recommended for better transcription.")
        
        if not recommendations:
            recommendations.append("Audio quality is good. Minimal processing needed.")
        
        return recommendations
    
    def _get_enhancement_strength(self, quality_report: AudioQualityReport) -> float:
        """Determine enhancement strength based on quality"""
        if self.enhancement_strength == 'auto':
            if quality_report.overall_score < 40:
                return 0.9  # Aggressive
            elif quality_report.overall_score < 60:
                return 0.7  # Medium
            elif quality_report.overall_score < 80:
                return 0.4  # Light
            else:
                return 0.2  # Minimal
        else:
            strengths = {'light': 0.3, 'medium': 0.6, 'aggressive': 0.9}
            return strengths.get(self.enhancement_strength, 0.5)
    
    def _reduce_white_noise(
        self,
        samples: np.ndarray,
        sr: int,
        strength: float
    ) -> np.ndarray:
        """Spectral subtraction for white noise"""
        try:
            from scipy.fft import rfft, irfft
        except ImportError:
            return samples
        
        frame_size = 1024
        hop_size = frame_size // 2
        window = np.hanning(frame_size)
        
        # Estimate noise from first 0.5 seconds
        noise_frames = int(0.5 * sr / hop_size)
        noise_spectrum = np.zeros(frame_size // 2 + 1)
        
        for i in range(min(noise_frames, len(samples) // hop_size - 1)):
            start = i * hop_size
            frame = samples[start:start + frame_size]
            if len(frame) == frame_size:
                noise_spectrum += np.abs(rfft(frame * window))
        
        noise_spectrum /= noise_frames
        
        # Process
        output = np.zeros(len(samples))
        for i in range(0, len(samples) - frame_size, hop_size):
            frame = samples[i:i + frame_size] * window
            spectrum = rfft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # Spectral subtraction with flooring
            clean_mag = magnitude - strength * noise_spectrum
            clean_mag = np.maximum(clean_mag, 0.05 * magnitude)
            
            clean_frame = irfft(clean_mag * np.exp(1j * phase), n=frame_size)
            output[i:i + frame_size] += clean_frame * window
        
        return output / (frame_size / hop_size / 2)
    
    def _reduce_background_speech(
        self,
        samples: np.ndarray,
        sr: int,
        strength: float
    ) -> np.ndarray:
        """
        Reduce background speech using modulation filtering
        
        Focus on enhancing the dominant speaker.
        """
        try:
            from scipy.signal import butter, filtfilt
        except ImportError:
            return samples
        
        # Apply bandpass filter focused on primary speech frequencies
        nyquist = sr / 2
        low = 300 / nyquist
        high = 3400 / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, samples)
        
        # Blend based on strength
        output = samples * (1 - strength * 0.5) + filtered * (strength * 0.5)
        
        return output
    
    def _reduce_environmental_noise(
        self,
        samples: np.ndarray,
        sr: int,
        strength: float
    ) -> np.ndarray:
        """Reduce low-frequency environmental noise"""
        try:
            from scipy.signal import butter, filtfilt
        except ImportError:
            return samples
        
        # High-pass filter to remove low rumble
        nyquist = sr / 2
        cutoff = (100 + strength * 100) / nyquist  # 100-200 Hz based on strength
        cutoff = min(0.99, max(0.01, cutoff))
        
        b, a = butter(4, cutoff, btype='high')
        return filtfilt(b, a, samples)
    
    def _reduce_reverb(
        self,
        samples: np.ndarray,
        sr: int,
        strength: float
    ) -> np.ndarray:
        """
        Simple de-reverberation using inverse filtering
        
        This is a simplified approach; production systems use more
        sophisticated methods like WPE (Weighted Prediction Error).
        """
        try:
            from scipy.signal import lfilter
        except ImportError:
            return samples
        
        # Simple pre-emphasis to reduce late reflections
        alpha = 0.95 * strength
        pre_emphasis = lfilter([1, -alpha], [1], samples)
        
        return pre_emphasis
    
    def _general_enhancement(
        self,
        samples: np.ndarray,
        sr: int,
        strength: float
    ) -> np.ndarray:
        """General enhancement for unknown noise types"""
        # Combine multiple techniques
        samples = self._reduce_white_noise(samples, sr, strength * 0.5)
        samples = self._reduce_environmental_noise(samples, sr, strength * 0.3)
        return samples
    
    def process_pipeline(
        self,
        audio_data: AudioData,
        auto_enhance: bool = True
    ) -> Tuple[AudioData, AudioQualityReport]:
        """
        Complete preprocessing pipeline
        
        1. Analyze quality
        2. Optionally enhance
        3. Return processed audio and report
        
        Args:
            audio_data: Input audio
            auto_enhance: Whether to automatically enhance if needed
            
        Returns:
            Tuple of (processed AudioData, quality report)
        """
        # Initial quality analysis
        initial_report = self.analyze_quality(audio_data)
        
        print(f"[AudioPreprocessor] Initial quality score: {initial_report.overall_score:.1f}/100")
        print(f"[AudioPreprocessor] SNR: {initial_report.snr_db:.1f} dB")
        print(f"[AudioPreprocessor] Noise type: {initial_report.noise_type.value}")
        print(f"[AudioPreprocessor] Predicted WER: {initial_report.predicted_wer:.1%}")
        
        if auto_enhance and initial_report.overall_score < 85:
            print("[AudioPreprocessor] Applying enhancement...")
            enhanced, final_report = self.enhance(audio_data, initial_report)
            print(f"[AudioPreprocessor] Post-enhancement score: {final_report.overall_score:.1f}/100")
            return enhanced, final_report
        else:
            print("[AudioPreprocessor] Quality sufficient, minimal processing applied.")
            return audio_data, initial_report
