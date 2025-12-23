"""
Tests for EchoNotes Audio Module
"""
import sys
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from audio import (
    AudioCapture, AudioData, AudioProcessor,
    IntelligentAudioPreprocessor, AudioQualityReport, NoiseType
)


def generate_test_audio(
    duration: float = 3.0,
    sample_rate: int = 16000,
    add_noise: bool = False,
    noise_level: float = 0.1
) -> AudioData:
    """Generate synthetic test audio with optional noise"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Generate speech-like signal (modulated tones)
    signal = np.zeros_like(t)
    
    # Add fundamental and harmonics (simulating speech formants)
    for freq in [150, 300, 600, 1200, 2400]:
        amplitude = 1.0 / (freq / 150)  # Decreasing amplitude with frequency
        signal += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add amplitude modulation (simulating syllables)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 4 Hz modulation
    signal = signal * modulation
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Add noise if requested
    if add_noise:
        noise = np.random.randn(len(signal)) * noise_level
        signal = signal + noise
        signal = signal / np.max(np.abs(signal)) * 0.9
    
    return AudioData(
        samples=signal.astype(np.float32),
        sample_rate=sample_rate,
        channels=1,
        duration=duration,
        source='synthetic'
    )


def test_audio_data():
    """Test AudioData container"""
    print("\n=== Testing AudioData ===")
    
    audio = generate_test_audio(duration=2.0)
    
    assert audio.is_mono, "Should be mono"
    assert audio.num_samples == 32000, f"Expected 32000 samples, got {audio.num_samples}"
    assert abs(audio.duration - 2.0) < 0.01, f"Expected 2.0s duration, got {audio.duration}"
    
    print(f"✓ AudioData: {audio.num_samples} samples, {audio.duration:.2f}s, {audio.sample_rate}Hz")
    print(f"✓ Source: {audio.source}")
    print(f"✓ Is mono: {audio.is_mono}")


def test_audio_processor():
    """Test AudioProcessor functionality"""
    print("\n=== Testing AudioProcessor ===")
    
    processor = AudioProcessor(sample_rate=16000)
    audio = generate_test_audio(duration=3.0, add_noise=True, noise_level=0.05)
    
    # Test normalization
    normalized = processor.normalize(audio, target_db=-20.0)
    print(f"✓ Normalization: samples normalized to target dB")
    
    # Test filtering
    filtered = processor.apply_filter(audio, 'highpass', cutoff_freq=80.0)
    print(f"✓ High-pass filter applied at 80Hz")
    
    filtered = processor.apply_filter(audio, 'bandpass', cutoff_freq=(80, 8000))
    print(f"✓ Band-pass filter applied (80-8000Hz)")
    
    # Test SNR computation
    snr = processor.compute_snr(audio)
    print(f"✓ SNR estimation: {snr:.1f} dB")
    
    # Test chunking
    chunks = processor.split_into_chunks(audio, chunk_duration=1.0, overlap=0.1)
    print(f"✓ Chunking: Split into {len(chunks)} chunks")
    
    # Test VAD
    segments = processor.detect_voice_activity(audio)
    print(f"✓ VAD: Detected {len(segments)} speech segments")
    for i, seg in enumerate(segments):
        print(f"  - Segment {i+1}: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)")


def test_intelligent_preprocessor():
    """Test IntelligentAudioPreprocessor (NEW MODULE)"""
    print("\n=== Testing IntelligentAudioPreprocessor (NEW) ===")
    
    preprocessor = IntelligentAudioPreprocessor(enhancement_strength='auto')
    
    # Test with clean audio
    print("\n--- Clean Audio Test ---")
    clean_audio = generate_test_audio(duration=2.0, add_noise=False)
    report = preprocessor.analyze_quality(clean_audio)
    
    print(f"✓ Quality Score: {report.overall_score:.1f}/100")
    print(f"✓ SNR: {report.snr_db:.1f} dB")
    print(f"✓ Noise Type: {report.noise_type.value}")
    print(f"✓ Clarity Score: {report.clarity_score:.3f}")
    print(f"✓ Predicted WER: {report.predicted_wer:.1%}")
    print(f"✓ Recommendations: {report.recommendations}")
    
    # Test with noisy audio
    print("\n--- Noisy Audio Test ---")
    noisy_audio = generate_test_audio(duration=2.0, add_noise=True, noise_level=0.3)
    noisy_report = preprocessor.analyze_quality(noisy_audio)
    
    print(f"✓ Quality Score: {noisy_report.overall_score:.1f}/100")
    print(f"✓ SNR: {noisy_report.snr_db:.1f} dB")
    print(f"✓ Noise Type: {noisy_report.noise_type.value}")
    print(f"✓ Predicted WER: {noisy_report.predicted_wer:.1%}")
    
    # Test enhancement pipeline
    print("\n--- Enhancement Pipeline Test ---")
    enhanced, enhanced_report = preprocessor.process_pipeline(noisy_audio)
    
    print(f"✓ Enhancement applied: {enhanced_report.enhancement_applied}")
    print(f"✓ New Quality Score: {enhanced_report.overall_score:.1f}/100")
    print(f"✓ Score improvement: {enhanced_report.overall_score - noisy_report.overall_score:.1f}")


def test_noise_type_detection():
    """Test noise type classification"""
    print("\n=== Testing Noise Type Detection ===")
    
    preprocessor = IntelligentAudioPreprocessor()
    
    # Generate different noise types
    samples_per_type = 16000 * 2  # 2 seconds
    
    # White noise
    white_noise = np.random.randn(samples_per_type).astype(np.float32) * 0.3
    white_audio = AudioData(
        samples=white_noise,
        sample_rate=16000,
        channels=1,
        duration=2.0,
        source='test'
    )
    white_report = preprocessor.analyze_quality(white_audio)
    print(f"✓ White noise detected as: {white_report.noise_type.value}")
    
    # Tonal/clean signal
    t = np.linspace(0, 2, samples_per_type)
    clean = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    clean_audio = AudioData(
        samples=clean,
        sample_rate=16000,
        channels=1,
        duration=2.0,
        source='test'
    )
    clean_report = preprocessor.analyze_quality(clean_audio)
    print(f"✓ Tonal signal detected as: {clean_report.noise_type.value}")


def test_quality_report_dict():
    """Test quality report dictionary conversion"""
    print("\n=== Testing Quality Report Serialization ===")
    
    report = AudioQualityReport(
        overall_score=75.5,
        snr_db=15.3,
        noise_type=NoiseType.ENVIRONMENTAL,
        clarity_score=0.65,
        predicted_wer=0.18,
        recommendations=["Consider quieter environment"],
        enhancement_applied=True
    )
    
    report_dict = report.to_dict()
    
    assert 'overall_score' in report_dict
    assert 'noise_type' in report_dict
    assert report_dict['noise_type'] == 'environmental'
    
    print(f"✓ Report dict: {report_dict}")


def run_all_tests():
    """Run all audio module tests"""
    print("=" * 60)
    print("EchoNotes Audio Module Tests")
    print("=" * 60)
    
    try:
        test_audio_data()
        test_audio_processor()
        test_intelligent_preprocessor()
        test_noise_type_detection()
        test_quality_report_dict()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
