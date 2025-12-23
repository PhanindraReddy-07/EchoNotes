#!/usr/bin/env python3
"""
EchoNotes Module 1 Demo
=======================
Run this script to test the Audio Input Layer

Usage:
    python demo_audio.py                    # Run with synthetic audio
    python demo_audio.py path/to/audio.wav  # Run with your audio file
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from audio import (
    AudioCapture,
    AudioProcessor, 
    IntelligentAudioPreprocessor,
    AudioQualityReport
)


def generate_demo_audio(duration: float = 5.0, sample_rate: int = 16000, noisy: bool = True):
    """Generate synthetic speech-like audio for demo"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Simulate speech with multiple formants
    signal = np.zeros_like(t)
    for freq in [150, 300, 600, 1200, 2400]:
        signal += (1.0 / (freq / 150)) * np.sin(2 * np.pi * freq * t)
    
    # Add syllable-like modulation
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    signal = signal * modulation
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.7
    
    # Add noise if requested
    if noisy:
        noise = np.random.randn(len(signal)) * 0.15
        signal = signal + noise
        signal = signal / np.max(np.abs(signal)) * 0.9
    
    from audio import AudioData
    return AudioData(
        samples=signal.astype(np.float32),
        sample_rate=sample_rate,
        channels=1,
        duration=duration,
        source='demo_synthetic'
    )


def print_header(text: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_report(report: AudioQualityReport):
    """Pretty print quality report"""
    print(f"""
┌─────────────────────────────────────────────────────────┐
│                 AUDIO QUALITY REPORT                     │
├─────────────────────────────────────────────────────────┤
│  Overall Score:    {report.overall_score:>6.1f} / 100                      │
│  SNR:              {report.snr_db:>6.1f} dB                           │
│  Noise Type:       {report.noise_type.value:<20}              │
│  Clarity Score:    {report.clarity_score:>6.3f}                          │
│  Predicted WER:    {report.predicted_wer:>6.1%}                          │
│  Enhanced:         {'Yes' if report.enhancement_applied else 'No':<5}                              │
├─────────────────────────────────────────────────────────┤
│  Recommendations:                                        │""")
    for rec in report.recommendations:
        # Wrap long recommendations
        if len(rec) > 55:
            print(f"│  • {rec[:52]}...│")
        else:
            print(f"│  • {rec:<55}│")
    print("└─────────────────────────────────────────────────────────┘")


def demo_with_file(filepath: str):
    """Demo with an actual audio file"""
    print_header("LOADING AUDIO FILE")
    
    capture = AudioCapture(target_sample_rate=16000, target_channels=1)
    
    try:
        audio = capture.load_file(filepath)
        print(f"✓ Loaded: {filepath}")
        print(f"  Duration: {audio.duration:.2f} seconds")
        print(f"  Sample Rate: {audio.sample_rate} Hz")
        print(f"  Channels: {audio.channels}")
        print(f"  Samples: {audio.num_samples:,}")
        return audio
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


def demo_with_synthetic():
    """Demo with synthetic audio"""
    print_header("GENERATING SYNTHETIC AUDIO")
    
    print("Creating 5-second synthetic speech signal with noise...")
    audio = generate_demo_audio(duration=5.0, noisy=True)
    
    print(f"✓ Generated synthetic audio")
    print(f"  Duration: {audio.duration:.2f} seconds")
    print(f"  Sample Rate: {audio.sample_rate} Hz")
    print(f"  Samples: {audio.num_samples:,}")
    
    return audio


def demo_processor(audio):
    """Demo AudioProcessor features"""
    print_header("AUDIO PROCESSOR DEMO")
    
    processor = AudioProcessor(sample_rate=audio.sample_rate)
    
    # SNR
    snr = processor.compute_snr(audio)
    print(f"✓ SNR Estimation: {snr:.1f} dB")
    
    # Normalization
    normalized = processor.normalize(audio, target_db=-20.0)
    print(f"✓ Normalized to -20 dB")
    
    # Filtering
    filtered = processor.apply_filter(audio, 'highpass', cutoff_freq=80.0)
    print(f"✓ Applied high-pass filter at 80 Hz")
    
    filtered = processor.apply_filter(audio, 'bandpass', cutoff_freq=(80, 8000))
    print(f"✓ Applied band-pass filter (80-8000 Hz)")
    
    # VAD
    segments = processor.detect_voice_activity(audio)
    print(f"✓ Voice Activity Detection: {len(segments)} segments found")
    for i, seg in enumerate(segments[:3]):  # Show first 3
        print(f"    Segment {i+1}: {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)")
    if len(segments) > 3:
        print(f"    ... and {len(segments) - 3} more")
    
    # Chunking
    chunks = processor.split_into_chunks(audio, chunk_duration=2.0, overlap=0.5)
    print(f"✓ Split into {len(chunks)} chunks (2s each, 0.5s overlap)")
    
    return normalized


def demo_intelligent_preprocessor(audio):
    """Demo the NEW IntelligentAudioPreprocessor"""
    print_header("INTELLIGENT AUDIO PREPROCESSOR (NEW)")
    
    preprocessor = IntelligentAudioPreprocessor(enhancement_strength='auto')
    
    # Initial analysis
    print("\n[1] Analyzing audio quality...")
    initial_report = preprocessor.analyze_quality(audio)
    print_report(initial_report)
    
    # Enhancement
    if initial_report.overall_score < 85:
        print("\n[2] Applying intelligent enhancement...")
        enhanced, final_report = preprocessor.enhance(audio, initial_report)
        
        improvement = final_report.overall_score - initial_report.overall_score
        print(f"\n✓ Enhancement complete!")
        print(f"  Score change: {initial_report.overall_score:.1f} → {final_report.overall_score:.1f} ({improvement:+.1f})")
        print(f"  WER prediction: {initial_report.predicted_wer:.1%} → {final_report.predicted_wer:.1%}")
        
        return enhanced, final_report
    else:
        print("\n[2] Audio quality is already good, skipping enhancement")
        return audio, initial_report


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ███████╗ ██████╗██╗  ██╗ ██████╗                     ║
    ║     ██╔════╝██╔════╝██║  ██║██╔═══██╗                    ║
    ║     █████╗  ██║     ███████║██║   ██║                    ║
    ║     ██╔══╝  ██║     ██╔══██║██║   ██║                    ║
    ║     ███████╗╚██████╗██║  ██║╚██████╔╝                    ║
    ║     ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝                    ║
    ║                                                           ║
    ║     N O T E S   -   Module 1: Audio Layer Demo           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check for file argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        audio = demo_with_file(filepath)
        if audio is None:
            print("\nFalling back to synthetic audio...")
            audio = demo_with_synthetic()
    else:
        audio = demo_with_synthetic()
    
    # Demo processor
    processed = demo_processor(audio)
    
    # Demo intelligent preprocessor
    enhanced, report = demo_intelligent_preprocessor(audio)
    
    # Summary
    print_header("DEMO COMPLETE")
    print("""
Module 1 Components Demonstrated:
  ✓ AudioCapture     - Load audio from files or generate synthetic
  ✓ AudioProcessor   - Filtering, normalization, VAD, chunking
  ✓ IntelligentAudioPreprocessor (NEW)
      - Noise type detection
      - Quality scoring
      - WER prediction
      - Adaptive enhancement

Next Steps:
  1. Try with your own audio file:
     python demo_audio.py your_audio.wav
  
  2. Run the test suite:
     python tests/test_audio.py
  
  3. Ready for Module 2 (Speech Recognition)?
     Just ask for the next module!
""")


if __name__ == "__main__":
    main()
