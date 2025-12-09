# EchoNotes - Windows/VS Code Version

Offline speech-to-document system optimized for Windows development with VS Code.

## Quick Start

### 1. Open in VS Code

```bash
cd echonotes-windows
code .
```

### 2. Setup (One-time)

**Option A: Use VS Code Tasks**
- Press `Ctrl+Shift+P` → "Tasks: Run Task" → "Full Setup (All Steps)"

**Option B: Manual PowerShell**
```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (needs internet - ~100MB for small)
python setup_models.py --small
```

### 3. Run EchoNotes

**From VS Code:**
- Press `F5` and select a launch configuration
- Or use `Ctrl+Shift+P` → "Tasks: Run Task"

**From Terminal:**
```powershell
# Activate environment
.venv\Scripts\activate

# Process audio file
python main.py --input samples\test.wav --output output\notes.md

# Record from microphone (30 seconds)
python main.py --record --duration 30 --output output\meeting.md

# Real-time streaming
python main.py --stream

# Process existing transcript
python main.py --text transcript.txt --output output\summary.md
```

## Project Structure

```
echonotes-windows/
├── .vscode/
│   ├── settings.json    # VS Code settings
│   ├── launch.json      # Debug configurations
│   └── tasks.json       # Build/run tasks
├── config/
│   ├── config.yaml      # Main configuration
│   └── settings.py      # Settings loader
├── modules/
│   ├── speech_recognition/
│   │   ├── audio_capture.py    # Mic/file input
│   │   ├── audio_processor.py  # Audio preprocessing
│   │   └── transcriber.py      # Vosk STT
│   ├── nlp_processor/
│   │   ├── preprocessor.py     # Text cleaning
│   │   ├── summarizer.py       # Summarization
│   │   ├── entity_extractor.py # NER
│   │   └── document_structurer.py
│   └── document_generator/
│       ├── markdown_gen.py
│       ├── txt_gen.py
│       └── json_gen.py
├── models/              # Downloaded models (gitignore)
├── output/              # Generated documents
├── samples/             # Test audio files
├── tests/               # Unit tests
├── main.py              # CLI entry point
├── setup_models.py      # Model downloader
└── requirements.txt
```

## VS Code Features

### Debug Configurations (F5)
- **Process Audio File** - Debug with sample audio
- **Record Mode** - Debug microphone recording
- **Stream Mode** - Debug real-time processing
- **Process Text** - Debug text-only processing
- **Run Tests** - Debug test suite

### Tasks (Ctrl+Shift+P → Tasks)
- **Full Setup** - Complete environment setup
- **Install Dependencies** - pip install
- **Download Models** - Setup models
- **Run Tests** - Execute test suite

### Recommended Extensions
- Python (Microsoft)
- Pylance
- Black Formatter
- YAML

## Configuration

Edit `config/config.yaml`:

```yaml
audio:
  sample_rate: 16000
  channels: 1

speech:
  model_size: small  # small, medium, large

nlp:
  use_gpu: false     # true if NVIDIA GPU available
  use_extractive_fallback: true

performance:
  low_memory_mode: false  # true for <8GB RAM
  batch_size: 4
```

## Output Formats

| Extension | Description |
|-----------|-------------|
| `.md` | Formatted Markdown with sections |
| `.txt` | Plain text |
| `.json` | Structured JSON data |

## Troubleshooting

### Audio device not found
```powershell
# List available devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Out of memory
1. Edit `config/config.yaml`:
   ```yaml
   performance:
     low_memory_mode: true
     batch_size: 1
   ```
2. Use small models: `python setup_models.py --small`

### Vosk model not loading
- Ensure model is extracted to `models/vosk-model-small/`
- Check the folder contains `am/`, `conf/`, `graph/` subdirectories

### CUDA/GPU errors
- Set `use_gpu: false` in config
- Or install CUDA toolkit matching your PyTorch version

## Testing

```powershell
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=modules --cov-report=html
```

## License

MIT License
