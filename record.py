#!/usr/bin/env python3
"""
EchoNotes Audio Recorder CLI
=============================
Record audio from microphone and process it

Usage:
    python record.py                     # Record for 10 seconds (default)
    python record.py --duration 30       # Record for 30 seconds
    python record.py --duration 60 --output my_recording.wav
    python record.py --list-devices      # Show available microphones
    python record.py --device 1          # Use specific microphone
"""

import sys
import argparse
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    
    try:
        import sounddevice
    except ImportError:
        missing.append('sounddevice')
    
    try:
        import soundfile
    except ImportError:
        missing.append('soundfile')
    
    if missing:
        print("❌ Missing dependencies!")
        print(f"   Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def list_devices():
    """List available audio input devices"""
    import sounddevice as sd
    
    print("\n🎤 Available Audio Input Devices:")
    print("=" * 50)
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append(i)
            default = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default}")
            print(f"      Channels: {dev['max_input_channels']}, Sample Rate: {int(dev['default_samplerate'])} Hz")
    
    if not input_devices:
        print("  ❌ No input devices found!")
    
    print("=" * 50)
    print(f"\nUse --device <number> to select a specific device")
    return input_devices


def record_audio(duration: float, device: int = None, sample_rate: int = 16000):
    """Record audio from microphone"""
    import sounddevice as sd
    import numpy as np
    
    print(f"\n🎙️  Recording for {duration} seconds...")
    print("   Speak now!")
    print()
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...", end='\r')
        time.sleep(1)
    
    print("   🔴 RECORDING...        ")
    print()
    
    # Progress bar during recording
    num_samples = int(duration * sample_rate)
    recording = sd.rec(
        num_samples,
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=device
    )
    
    # Show progress
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = int((elapsed / duration) * 40)
        bar = "█" * progress + "░" * (40 - progress)
        print(f"   [{bar}] {elapsed:.1f}s / {duration:.1f}s", end='\r')
        time.sleep(0.1)
    
    sd.wait()  # Wait until recording is finished
    
    print(f"   [{'█' * 40}] {duration:.1f}s / {duration:.1f}s")
    print("\n   ✅ Recording complete!")
    
    return recording.flatten(), sample_rate


def save_audio(samples, sample_rate: int, filepath: str):
    """Save audio to file"""
    import soundfile as sf
    
    sf.write(filepath, samples, sample_rate)
    print(f"   💾 Saved to: {filepath}")
    return filepath


def process_audio(samples, sample_rate: int):
    """Process recorded audio through Module 1 pipeline"""
    from audio import AudioData, AudioProcessor, IntelligentAudioPreprocessor
    
    # Create AudioData object
    audio = AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=1,
        duration=len(samples) / sample_rate,
        source='microphone'
    )
    
    print("\n" + "=" * 50)
    print("  PROCESSING AUDIO")
    print("=" * 50)
    
    # Basic processing
    processor = AudioProcessor(sample_rate=sample_rate)
    
    # SNR
    snr = processor.compute_snr(audio)
    print(f"\n  📊 Initial SNR: {snr:.1f} dB")
    
    # VAD
    segments = processor.detect_voice_activity(audio)
    total_speech = sum(seg.duration for seg in segments)
    print(f"  🗣️  Speech detected: {total_speech:.1f}s / {audio.duration:.1f}s")
    
    if segments:
        print(f"     Found {len(segments)} speech segment(s):")
        for i, seg in enumerate(segments[:5]):
            print(f"       • {seg.start_time:.2f}s - {seg.end_time:.2f}s ({seg.duration:.2f}s)")
        if len(segments) > 5:
            print(f"       ... and {len(segments) - 5} more")
    
    # Intelligent preprocessing
    print("\n  🧠 Analyzing audio quality...")
    preprocessor = IntelligentAudioPreprocessor()
    report = preprocessor.analyze_quality(audio)
    
    print(f"""
  ┌────────────────────────────────────────────┐
  │           QUALITY REPORT                   │
  ├────────────────────────────────────────────┤
  │  Overall Score:   {report.overall_score:>5.1f} / 100             │
  │  SNR:             {report.snr_db:>5.1f} dB                │
  │  Noise Type:      {report.noise_type.value:<20}   │
  │  Clarity:         {report.clarity_score:>5.3f}                  │
  │  Predicted WER:   {report.predicted_wer:>5.1%}                  │
  └────────────────────────────────────────────┘
    """)
    
    print("  📝 Recommendations:")
    for rec in report.recommendations:
        print(f"     • {rec}")
    
    # Enhancement if needed
    if report.overall_score < 80:
        print("\n  🔧 Applying enhancement...")
        enhanced, new_report = preprocessor.enhance(audio, report)
        print(f"     Score: {report.overall_score:.1f} → {new_report.overall_score:.1f}")
        print(f"     WER:   {report.predicted_wer:.1%} → {new_report.predicted_wer:.1%}")
        return enhanced
    
    return audio


def main():
    parser = argparse.ArgumentParser(
        description="EchoNotes Audio Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python record.py                          # Record 10 seconds
  python record.py -d 30                    # Record 30 seconds
  python record.py -d 60 -o meeting.wav     # Record 60s, save as meeting.wav
  python record.py --list-devices           # Show microphones
  python record.py --device 2 -d 15         # Use device #2, record 15s
        """
    )
    
    parser.add_argument('-d', '--duration', type=float, default=10,
                        help='Recording duration in seconds (default: 10)')
    parser.add_argument('-o', '--output', type=str, default='recording.wav',
                        help='Output filename (default: recording.wav)')
    parser.add_argument('--device', type=int, default=None,
                        help='Audio device index (use --list-devices to see options)')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio input devices')
    parser.add_argument('--no-process', action='store_true',
                        help='Skip audio processing (just record and save)')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Sample rate in Hz (default: 16000)')
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    print("""
    ╔═══════════════════════════════════════════════╗
    ║     🎙️  EchoNotes Audio Recorder              ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # List devices if requested
    if args.list_devices:
        list_devices()
        return
    
    # Show selected device
    if args.device is not None:
        import sounddevice as sd
        devices = sd.query_devices()
        if args.device < len(devices):
            print(f"  Using device [{args.device}]: {devices[args.device]['name']}")
        else:
            print(f"  ❌ Device {args.device} not found!")
            list_devices()
            return
    else:
        import sounddevice as sd
        default_device = sd.default.device[0]
        devices = sd.query_devices()
        print(f"  Using default device [{default_device}]: {devices[default_device]['name']}")
    
    print(f"  Duration: {args.duration} seconds")
    print(f"  Output: {args.output}")
    
    try:
        # Record
        samples, sr = record_audio(
            duration=args.duration,
            device=args.device,
            sample_rate=args.sample_rate
        )
        
        # Save
        save_audio(samples, sr, args.output)
        
        # Process (unless skipped)
        if not args.no_process:
            process_audio(samples, sr)
        
        print("\n" + "=" * 50)
        print("  ✅ DONE!")
        print("=" * 50)
        print(f"\n  Your recording: {args.output}")
        print(f"  Process with:   python demo_audio.py {args.output}")
        print()
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Recording cancelled by user")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
