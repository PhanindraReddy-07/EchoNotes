# EchoNotes - Module 1 File Structure

```
echonotes/
│
├── __init__.py                 # Main package init (49 lines)
├── README.md                   # Full documentation (253 lines)
├── requirements.txt            # Dependencies (42 lines)
├── demo_audio.py               # 🎯 RUN THIS TO TEST! (200+ lines)
│
├── config/                     # Configuration module
│   ├── __init__.py            # Config exports (23 lines)
│   └── settings.py            # All settings dataclasses (162 lines)
│       ├── AudioConfig        # Audio processing settings
│       ├── ASRConfig          # Speech recognition settings
│       ├── DiarizationConfig  # Speaker diarization settings
│       ├── NLPConfig          # NLP processing settings
│       ├── OutputConfig       # Document output settings
│       ├── SystemConfig       # System-wide settings
│       └── EchoNotesConfig    # Main config combining all
│
├── audio/                      # 🎤 AUDIO MODULE (Module 1)
│   ├── __init__.py            # Module exports (58 lines)
│   ├── capture.py             # Audio acquisition (319 lines)
│   │   ├── AudioData          # Audio container dataclass
│   │   └── AudioCapture       # Multi-source capture class
│   │       ├── load_file()    # Load WAV/MP3/FLAC/OGG/M4A
│   │       ├── load_bytes()   # Load from bytes (web upload)
│   │       ├── record_microphone()  # Record from mic
│   │       ├── stream_microphone()  # Stream from mic
│   │       └── list_devices() # List audio devices
│   │
│   ├── processor.py           # Signal processing (429 lines)
│   │   ├── SpeechSegment      # VAD segment dataclass
│   │   └── AudioProcessor     # Processing class
│   │       ├── normalize()    # Normalize to target dB
│   │       ├── apply_filter() # High/low/band-pass filters
│   │       ├── reduce_noise_simple()  # Basic noise reduction
│   │       ├── detect_voice_activity()  # VAD
│   │       ├── compute_snr()  # Signal-to-noise ratio
│   │       └── split_into_chunks()  # Chunk for processing
│   │
│   └── enhancer.py            # ⭐ NEW: Intelligent preprocessing (601 lines)
│       ├── NoiseType          # Enum: CLEAN, WHITE_NOISE, REVERB, etc.
│       ├── AudioQualityReport # Quality analysis dataclass
│       └── IntelligentAudioPreprocessor  # Main class
│           ├── analyze_quality()   # Full quality analysis
│           ├── enhance()           # Apply enhancement
│           ├── process_pipeline()  # Complete pipeline
│           ├── _detect_noise_type()     # Classify noise
│           ├── _compute_clarity_score() # Speech clarity
│           ├── _predict_wer()           # Predict Word Error Rate
│           ├── _reduce_white_noise()    # White noise reduction
│           ├── _reduce_reverb()         # Reverb reduction
│           └── _reduce_environmental_noise()  # Low-freq noise
│
├── tests/                      # Test suite
│   └── test_audio.py          # Audio module tests (226 lines)
│
├── speech/                     # (Module 2 - Coming Next)
├── nlp/                        # (Module 3 - Coming Soon)
├── document/                   # (Module 4 - Coming Soon)
├── api/                        # (Module 5 - Coming Soon)
├── evaluation/                 # (Evaluation metrics)
└── models/                     # (Downloaded ML models)
```

## 🚀 How to Run

### Option 1: Run the Demo Script (Recommended)
```bash
cd echonotes
python demo_audio.py
```

### Option 2: Run with Your Audio File
```bash
python demo_audio.py path/to/your/audio.wav
```

### Option 3: Run Tests
```bash
python tests/test_audio.py
```

### Option 4: Use in Your Code
```python
from audio import AudioCapture, AudioProcessor, IntelligentAudioPreprocessor

# Load audio
capture = AudioCapture(target_sample_rate=16000)
audio = capture.load_file("meeting.wav")

# Process
processor = AudioProcessor()
normalized = processor.normalize(audio)

# Enhance (NEW feature!)
enhancer = IntelligentAudioPreprocessor()
enhanced, report = enhancer.process_pipeline(audio)
print(f"Quality: {report.overall_score}/100")
```

## 📦 Dependencies Required

```bash
pip install numpy scipy soundfile
```

Optional (for microphone recording):
```bash
pip install sounddevice
```

## 📊 Module 1 Statistics

| Component | Lines of Code | Classes | Methods |
|-----------|---------------|---------|---------|
| capture.py | 319 | 2 | 8 |
| processor.py | 429 | 2 | 9 |
| enhancer.py | 601 | 3 | 15 |
| settings.py | 162 | 7 | 3 |
| **Total** | **1,511** | **14** | **35** |

## ✅ Features Implemented

- [x] Multi-format audio loading (WAV, MP3, FLAC, OGG, M4A)
- [x] Microphone recording and streaming
- [x] Audio normalization
- [x] Frequency filtering (high-pass, low-pass, band-pass)
- [x] Voice Activity Detection (VAD)
- [x] SNR computation
- [x] Audio chunking for long files
- [x] **Noise type classification** (6 types)
- [x] **Quality scoring** (0-100)
- [x] **WER prediction**
- [x] **Adaptive noise reduction**
- [x] Comprehensive configuration system
