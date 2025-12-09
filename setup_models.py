"""
EchoNotes Model Setup Script
Downloads all required models for offline operation.

Usage:
    python setup_models.py          # Full models
    python setup_models.py --small  # Lightweight models
"""

import argparse
import os
import sys
import zipfile
import urllib.request
from pathlib import Path


# Vosk models
VOSK_MODELS = {
    'small': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
        'name': 'vosk-model-small',
        'size': '~50MB'
    },
    'medium': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
        'name': 'vosk-model-medium',
        'size': '~500MB'
    },
    'large': {
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip',
        'name': 'vosk-model-large',
        'size': '~1.8GB'
    }
}


def download_with_progress(url: str, dest: Path):
    """Download file with progress bar."""
    print(f"Downloading: {url}")
    
    def progress_hook(count, block_size, total_size):
        percent = min(100, count * block_size * 100 // total_size)
        bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
        sys.stdout.write(f'\r[{bar}] {percent}%')
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print()  # New line after progress


def setup_vosk(models_dir: Path, size: str = 'small'):
    """Download and extract Vosk model."""
    model_info = VOSK_MODELS.get(size, VOSK_MODELS['small'])
    
    print(f"\n=== Setting up Vosk ({model_info['size']}) ===")
    
    # Target directory
    model_dir = models_dir / f"vosk-model-{size}"
    
    if model_dir.exists():
        print(f"Vosk model already exists: {model_dir}")
        return
    
    # Download
    zip_path = models_dir / 'vosk_model.zip'
    download_with_progress(model_info['url'], zip_path)
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(models_dir)
    
    # Rename to standard name
    extracted = list(models_dir.glob('vosk-model-*'))
    for path in extracted:
        if path.is_dir() and path != model_dir:
            path.rename(model_dir)
            break
    
    # Cleanup
    zip_path.unlink()
    print(f"Vosk model ready: {model_dir}")


def setup_nltk():
    """Download NLTK data."""
    print("\n=== Setting up NLTK ===")
    
    try:
        import nltk
        
        packages = [
            'punkt',
            'punkt_tab',
            'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng',
            'maxent_ne_chunker',
            'maxent_ne_chunker_tab',
            'words',
            'stopwords',
            'wordnet'
        ]
        
        for package in packages:
            print(f"Downloading {package}...")
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                print(f"  Warning: {e}")
        
        print("NLTK data ready")
    except ImportError:
        print("NLTK not installed. Run: pip install nltk")


def setup_transformers(models_dir: Path, small: bool = False):
    """Download transformer models for offline use."""
    print("\n=== Setting up Transformers ===")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Choose model
        if small:
            model_name = 'google/flan-t5-small'
        else:
            model_name = 'sshleifer/distilbart-cnn-12-6'
        
        print(f"Downloading {model_name}...")
        
        save_dir = models_dir / 'summarizer'
        save_dir.mkdir(exist_ok=True)
        
        # Download and save locally
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        
        print(f"Transformer model saved: {save_dir}")
    except ImportError:
        print("Transformers not installed. Run: pip install transformers torch")
    except Exception as e:
        print(f"Warning: Could not download transformer model: {e}")
        print("Extractive summarization will be used instead.")


def main():
    parser = argparse.ArgumentParser(description='Setup EchoNotes models')
    parser.add_argument('--small', action='store_true', 
                        help='Use small models (lower accuracy, faster)')
    parser.add_argument('--large', action='store_true',
                        help='Use large models (best accuracy, slower)')
    parser.add_argument('--vosk-only', action='store_true',
                        help='Only download Vosk model')
    parser.add_argument('--nltk-only', action='store_true',
                        help='Only download NLTK data')
    parser.add_argument('--transformers-only', action='store_true',
                        help='Only download transformer model')
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent
    models_dir = project_root / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Determine model size
    if args.small:
        vosk_size = 'small'
        mode_name = 'Small (fast, lower accuracy)'
    elif args.large:
        vosk_size = 'large'
        mode_name = 'Large (best accuracy, slowest)'
    else:
        vosk_size = 'medium'  # Default to medium for better accuracy
        mode_name = 'Medium (recommended, good accuracy)'
    
    print("EchoNotes Model Setup")
    print("=" * 40)
    print(f"Models directory: {models_dir}")
    print(f"Mode: {mode_name}")
    
    # Determine what to download
    download_all = not (args.vosk_only or args.nltk_only or args.transformers_only)
    
    try:
        if download_all or args.vosk_only:
            setup_vosk(models_dir, vosk_size)
        
        if download_all or args.nltk_only:
            setup_nltk()
        
        if download_all or args.transformers_only:
            setup_transformers(models_dir, args.small)
        
        # Update config to match downloaded model
        update_config(project_root, vosk_size)
        
        print("\n" + "=" * 40)
        print("Setup complete!")
        print("\nYou can now run EchoNotes offline:")
        print("  python main.py --input audio.wav --output notes.md")
        print("  python main.py --record --duration 30")
    
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)


def update_config(project_root: Path, vosk_size: str):
    """Update config.yaml with the downloaded model size."""
    config_path = project_root / 'config' / 'config.yaml'
    if config_path.exists():
        content = config_path.read_text()
        # Update model_size in config
        import re
        content = re.sub(
            r'model_size:\s*\w+',
            f'model_size: {vosk_size}',
            content
        )
        config_path.write_text(content)
        print(f"\nUpdated config.yaml: model_size = {vosk_size}")


if __name__ == '__main__':
    main()