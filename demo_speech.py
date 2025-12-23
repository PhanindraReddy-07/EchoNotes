#!/usr/bin/env python3
"""
EchoNotes Module 2 Demo - Speech Recognition
=============================================

Usage:
    python demo_speech.py recording.wav                    # Transcribe audio file
    python demo_speech.py recording.wav --model path/to/model  # With specific model
    python demo_speech.py --download-model                 # Download Vosk model first

Prerequisites:
    pip install vosk resemblyzer scikit-learn
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def download_model(model_name: str = 'en-us'):
    """Download a Vosk model"""
    from speech import Transcriber
    
    model_sizes = {
        'en-us': '~1.8 GB (best accuracy - 5.7% WER) ⭐',
        'en-us-lgraph': '~128 MB (medium accuracy - 7.8% WER)',
        'en-us-small': '~40 MB (fast, lower accuracy - 9.8% WER)',
        'en-in': '~1 GB (Indian English accent)',
        'en-in-small': '~36 MB (Indian English, lightweight)',
        'hi': '~1.5 GB (Hindi)',
        'hi-small': '~42 MB (Hindi lightweight)',
        'te-small': '~58 MB (Telugu)',
    }
    
    print(f"\n📥 Downloading Vosk model: {model_name}")
    if model_name in model_sizes:
        print(f"   Size: {model_sizes[model_name]}")
    print("   This may take several minutes...\n")
    
    model_path = Transcriber.download_model(model_name, output_dir="./models")
    
    print(f"\n✅ Model downloaded to: {model_path}")
    print(f"   Use with: python demo_speech.py audio.wav --model {model_path}")
    
    return model_path


def find_model():
    """Try to find an existing Vosk model"""
    model_dirs = [
        Path("./models"),
        Path("./vosk-model-en-us-0.22"),
        Path("./vosk-model-small-en-us-0.15"),
        Path.home() / "vosk-models",
    ]
    
    for base_dir in model_dirs:
        if base_dir.exists():
            # Look for model directories
            for item in base_dir.iterdir():
                if item.is_dir() and 'vosk' in item.name.lower():
                    return str(item)
            # Check if base_dir itself is a model
            if (base_dir / "am" / "final.mdl").exists():
                return str(base_dir)
    
    return None


def demo_transcription(audio_path: str, model_path: str):
    """Demo transcription with Vosk"""
    from audio import AudioCapture, IntelligentAudioPreprocessor
    from speech import Transcriber, ConfidenceScorer, format_confidence_report
    
    print("\n" + "=" * 60)
    print("  LOADING AUDIO")
    print("=" * 60)
    
    # Load and preprocess audio
    capture = AudioCapture(target_sample_rate=16000)
    audio = capture.load_file(audio_path)
    
    print(f"  ✓ Loaded: {audio_path}")
    print(f"  ✓ Duration: {audio.duration:.1f}s")
    
    # Preprocess
    preprocessor = IntelligentAudioPreprocessor()
    enhanced, quality_report = preprocessor.process_pipeline(audio, auto_enhance=True)
    
    print("\n" + "=" * 60)
    print("  TRANSCRIBING")
    print("=" * 60)
    
    # Transcribe
    transcriber = Transcriber(model_path=model_path)
    result = transcriber.transcribe(enhanced)
    
    print(f"\n  ✓ Transcription complete!")
    print(f"  ✓ Words: {result.word_count}")
    print(f"  ✓ Average confidence: {result.avg_confidence:.1%}")
    
    # Show transcript
    print("\n" + "=" * 60)
    print("  TRANSCRIPT")
    print("=" * 60)
    print()
    
    # Show with timestamps every 10 seconds
    timestamped = result.get_text_with_timestamps(interval=10.0)
    print(timestamped)
    
    # === SAVE TRANSCRIPT TO FILE ===
    audio_name = Path(audio_path).stem
    txt_path = f"{audio_name}_transcript.txt"
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"EchoNotes Transcript\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Audio: {audio_path}\n")
        f.write(f"Duration: {audio.duration:.1f}s\n")
        f.write(f"Words: {result.word_count}\n")
        f.write(f"Confidence: {result.avg_confidence:.1%}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"TRANSCRIPT:\n")
        f.write(f"{'-' * 50}\n")
        f.write(result.text)
        f.write(f"\n{'-' * 50}\n\n")
        f.write(f"WITH TIMESTAMPS:\n")
        f.write(timestamped)
        f.write(f"\n\n{'=' * 50}\n")
    
    print(f"\n  💾 Transcript saved to: {txt_path}")
    
    # Confidence analysis
    print("\n" + "=" * 60)
    print("  CONFIDENCE ANALYSIS")
    print("=" * 60)
    
    scorer = ConfidenceScorer()
    conf_report = scorer.analyze(result)
    print(format_confidence_report(conf_report))
    
    # Show low-confidence words highlighted
    if conf_report.low_confidence_spans:
        print("\n  📝 Text with uncertain words marked:")
        highlighted = scorer.get_highlighted_text(result, format='plain')
        print(f"  {highlighted[:500]}..." if len(highlighted) > 500 else f"  {highlighted}")
    
    return result


def demo_diarization(audio_path: str, transcription_result=None):
    """Demo speaker diarization"""
    from audio import AudioCapture
    from speech import SpeakerDiarizer
    
    print("\n" + "=" * 60)
    print("  SPEAKER DIARIZATION")
    print("=" * 60)
    
    # Load audio
    capture = AudioCapture(target_sample_rate=16000)
    audio = capture.load_file(audio_path)
    
    # Diarize
    print("\n  🔍 Identifying speakers...")
    
    try:
        diarizer = SpeakerDiarizer()
        result = diarizer.diarize(audio)
        
        print(f"\n  ✓ Found {result.num_speakers} speaker(s)")
        print(f"  ✓ {len(result.segments)} segments")
        
        # Show speaker stats
        print("\n  📊 Speaker Statistics:")
        for speaker, stats in result.speaker_stats.items():
            print(f"     {speaker}: {stats['total_time']:.1f}s ({stats['percentage']:.1f}%)")
        
        # Show timeline
        print("\n  📅 Timeline:")
        print(result.get_timeline())
        
        # Assign speakers to transcript if available
        if transcription_result:
            diarizer.assign_speakers_to_transcript(result, transcription_result)
            print("\n  ✓ Speakers assigned to transcript")
        
        return result
        
    except ImportError as e:
        print(f"\n  ⚠️  Diarization requires additional packages:")
        print(f"     pip install resemblyzer scikit-learn")
        print(f"     Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="EchoNotes Speech Recognition Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_speech.py --download-model              # Download BEST model (~1.8GB, 5.7%% WER) ⭐
  python demo_speech.py --download-small              # Download small model (~40MB, faster)
  python demo_speech.py --download-indian             # Download Indian English model
  python demo_speech.py recording.wav                 # Transcribe with auto-detected model
  python demo_speech.py recording.wav --model ./models/vosk-model-en-us-0.22
  python demo_speech.py recording.wav --no-diarize    # Skip speaker identification
        """
    )
    
    parser.add_argument('audio', nargs='?', help='Path to audio file')
    parser.add_argument('--model', '-m', help='Path to Vosk model directory')
    parser.add_argument('--download-model', action='store_true', 
                        help='Download best model (en-us, ~1.8GB, 5.7%% WER) ⭐')
    parser.add_argument('--download-small', action='store_true',
                        help='Download small model (en-us-small, ~40MB, faster)')
    parser.add_argument('--download-large', action='store_true',
                        help='Same as --download-model')
    parser.add_argument('--download-indian', action='store_true',
                        help='Download Indian English model (en-in, ~1GB)')
    parser.add_argument('--no-diarize', action='store_true',
                        help='Skip speaker diarization')
    
    args = parser.parse_args()
    
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
    ║     N O T E S   -   Module 2: Speech Recognition         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Handle model download
    if args.download_model:
        download_model('en-us')  # 1.8GB - BEST accuracy
        return
    
    if args.download_small:
        download_model('en-us-small')  # 40MB - fast
        return
    
    if args.download_large:
        download_model('en-us')  # Same as default
        return
    
    if args.download_indian:
        download_model('en-in')  # Indian English
        return
    
    # Check for audio file
    if not args.audio:
        print("  ❌ No audio file specified!")
        print()
        print("  Usage:")
        print("    1. First download a model:")
        print("       python demo_speech.py --download-model")
        print()
        print("    2. Then transcribe audio:")
        print("       python demo_speech.py your_audio.wav")
        print()
        print("  Or record audio first:")
        print("       python record.py --duration 30")
        print("       python demo_speech.py recording.wav")
        return
    
    # Check audio file exists
    if not Path(args.audio).exists():
        print(f"  ❌ Audio file not found: {args.audio}")
        return
    
    # Find or specify model
    model_path = args.model
    if not model_path:
        model_path = find_model()
        
        if not model_path:
            print("  ❌ No Vosk model found!")
            print()
            print("  Download a model first:")
            print()
            print("  ⭐ BEST ACCURACY (recommended):")
            print("    python demo_speech.py --download-model")
            print("    Size: ~1.8 GB | WER: 5.7% (best)")
            print()
            print("  ⚡ FAST (for testing):")
            print("    python demo_speech.py --download-small")
            print("    Size: ~40 MB | WER: 9.8%")
            print()
            print("  🇮🇳 INDIAN ENGLISH:")
            print("    python demo_speech.py --download-indian")
            print("    Size: ~1 GB | For Indian accent")
            return
        
        print(f"  📁 Found model: {model_path}")
    
    # Run demos
    try:
        # Transcription
        transcript = demo_transcription(args.audio, model_path)
        
        # Diarization (optional)
        if not args.no_diarize:
            demo_diarization(args.audio, transcript)
        
        print("\n" + "=" * 60)
        print("  ✅ DEMO COMPLETE")
        print("=" * 60)
        print("""
  Module 2 Components Demonstrated:
    ✓ Transcriber       - Offline speech-to-text (Vosk)
    ✓ ConfidenceScorer  - Word-level confidence analysis
    ✓ SpeakerDiarizer   - Speaker identification
  
  Next: Module 3 (NLP Processing) for:
    • Entity extraction (actions, decisions, deadlines)
    • Summarization
    • Text preprocessing
        """)
        
    except ImportError as e:
        print(f"\n  ❌ Missing dependency: {e}")
        print("\n  Install required packages:")
        print("    pip install vosk resemblyzer scikit-learn")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
