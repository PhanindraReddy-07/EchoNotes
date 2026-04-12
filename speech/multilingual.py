"""
EchoNotes - Multilingual Transcriber
======================================
Extends the existing Transcriber to support Indian English, Hindi, and Telugu.

Uses the same Word, Utterance, TranscriptionResult data classes from transcriber.py
so it plugs directly into the existing NLP and document generation pipeline.

Usage:
    from speech.multilingual import MultilingualTranscriber
    
    mt = MultilingualTranscriber(models_dir="./models")
    mt.download_model("hi")       # Download Hindi model
    mt.download_model("te")       # Download Telugu model
    
    result = mt.transcribe("lecture.wav", language="hi")
    result = mt.transcribe("lecture.wav", language="auto")
"""

import json
import wave
import os
import logging
import urllib.request
import zipfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .transcriber import Transcriber, Word, Utterance, TranscriptionResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Language & Model Configuration
# ─────────────────────────────────────────────

class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    TELUGU = "te"


# Official Vosk model registry for Indian languages
MODEL_REGISTRY = {
    # ── Indian English ──
    "en-in": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-in-0.5.zip",
        "dir_name": "vosk-model-en-in-0.5",
        "size": "~1 GB",
        "lang": Language.ENGLISH,
        "label": "Indian English",
        "type": "large",
    },
    "en-in-small": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",
        "dir_name": "vosk-model-small-en-in-0.4",
        "size": "~36 MB",
        "lang": Language.ENGLISH,
        "label": "Indian English (small)",
        "type": "small",
    },
    # ── Hindi ──
    "hi": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip",
        "dir_name": "vosk-model-hi-0.22",
        "size": "~1.5 GB",
        "lang": Language.HINDI,
        "label": "Hindi",
        "type": "large",
    },
    "hi-small": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
        "dir_name": "vosk-model-small-hi-0.22",
        "size": "~42 MB",
        "lang": Language.HINDI,
        "label": "Hindi (small)",
        "type": "small",
    },
    # ── Telugu ──
    "te": {
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-te-0.42.zip",
        "dir_name": "vosk-model-small-te-0.42",
        "size": "~58 MB",
        "lang": Language.TELUGU,
        "label": "Telugu",
        "type": "small",
    },
}

# Preferred model order per language (try large first, fall back to small)
PREFERRED_MODELS = {
    Language.ENGLISH: ["en-in", "en-in-small"],
    Language.HINDI: ["hi", "hi-small"],
    Language.TELUGU: ["te"],
}

# Map user-friendly strings to Language enum
LANGUAGE_MAP = {
    "en": Language.ENGLISH,
    "en-in": Language.ENGLISH,
    "english": Language.ENGLISH,
    "hi": Language.HINDI,
    "hindi": Language.HINDI,
    "te": Language.TELUGU,
    "telugu": Language.TELUGU,
}


class MultilingualTranscriber:
    """
    Manages multiple Vosk models for Indian English, Hindi, and Telugu.
    
    Returns the same TranscriptionResult as the existing Transcriber,
    so it integrates seamlessly with the NLP and document generation pipeline.
    """

    SAMPLE_RATE = 16000

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._vosk = None
        self._loaded_models: Dict[str, object] = {}  # model_key -> vosk.Model
        self._model_paths: Dict[str, Path] = {}       # model_key -> Path
        self._discover_models()

    def _get_vosk(self):
        """Lazy import Vosk."""
        if self._vosk is None:
            try:
                import vosk
                vosk.SetLogLevel(-1)
                self._vosk = vosk
            except ImportError:
                raise ImportError("Vosk is required: pip install vosk")
        return self._vosk

    # ─────────────────────────────────────────
    # Model Discovery & Download
    # ─────────────────────────────────────────

    def _discover_models(self):
        """Scan models directory for already-downloaded Vosk models."""
        self._model_paths.clear()
        if not self.models_dir.exists():
            return

        for entry in self.models_dir.iterdir():
            if not entry.is_dir():
                continue
            for model_key, info in MODEL_REGISTRY.items():
                if info["dir_name"] in entry.name:
                    self._model_paths[model_key] = entry
                    logger.info(f"Found model '{model_key}' at {entry}")

    def get_available_models(self) -> Dict[str, dict]:
        """Return info about all models with download status."""
        result = {}
        for key, info in MODEL_REGISTRY.items():
            result[key] = {
                **info,
                "downloaded": key in self._model_paths,
                "path": str(self._model_paths.get(key, "")),
            }
        return result

    def get_downloaded_models(self) -> List[str]:
        """Return list of downloaded model keys."""
        return list(self._model_paths.keys())

    def get_downloaded_languages(self) -> List[str]:
        """Return unique language codes that have downloaded models."""
        langs = set()
        for key in self._model_paths:
            langs.add(MODEL_REGISTRY[key]["lang"].value)
        return sorted(langs)

    def download_model(self, model_key: str, progress_callback=None) -> Path:
        """Download a Vosk model by key (e.g., 'hi', 'te', 'en-in')."""
        if model_key not in MODEL_REGISTRY:
            available = list(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available}")

        if model_key in self._model_paths:
            print(f"   ✅ Model '{model_key}' already downloaded at {self._model_paths[model_key]}")
            return self._model_paths[model_key]

        info = MODEL_REGISTRY[model_key]
        url = info["url"]
        zip_path = self.models_dir / f"{model_key}.zip"

        print(f"\n📥 Downloading {info['label']} model ({info['size']})...")
        print(f"   URL: {url}")

        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r   [{bar}] {pct}% ({downloaded // (1024*1024)}MB)", end="", flush=True)
            if progress_callback:
                progress_callback(downloaded, total_size)

        urllib.request.urlretrieve(url, zip_path, reporthook=_reporthook)
        print()

        print("   📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.models_dir)
        zip_path.unlink()

        model_path = self.models_dir / info["dir_name"]
        if not model_path.exists():
            for item in self.models_dir.iterdir():
                if item.is_dir() and info["dir_name"] in item.name:
                    model_path = item
                    break

        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found after extraction: {info['dir_name']}")

        self._model_paths[model_key] = model_path
        print(f"   ✅ Model ready at: {model_path}")
        return model_path

    def download_all(self, prefer_small: bool = False) -> Dict[str, Path]:
        """Download models for all three languages."""
        targets = {
            "en-in-small" if prefer_small else "en-in": "Indian English",
            "hi-small" if prefer_small else "hi": "Hindi",
            "te": "Telugu",
        }
        downloaded = {}
        for model_key, label in targets.items():
            try:
                path = self.download_model(model_key)
                downloaded[model_key] = path
            except Exception as e:
                print(f"   ❌ Failed to download {label}: {e}")
        return downloaded

    # ─────────────────────────────────────────
    # Model Loading
    # ─────────────────────────────────────────

    def _load_model(self, model_key: str):
        """Load a Vosk model into memory (cached)."""
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]

        if model_key not in self._model_paths:
            raise FileNotFoundError(
                f"Model '{model_key}' not downloaded. "
                f"Run: transcriber.download_model('{model_key}')"
            )

        vosk = self._get_vosk()
        path = self._model_paths[model_key]
        print(f"[MultilingualTranscriber] Loading model '{model_key}' from {path}...")
        model = vosk.Model(str(path))
        self._loaded_models[model_key] = model
        print(f"[MultilingualTranscriber] Model '{model_key}' loaded")
        return model

    def _unload_model(self, model_key: str):
        """Unload a model from memory."""
        self._loaded_models.pop(model_key, None)

    def _get_best_model_for_language(self, lang: Language) -> Optional[str]:
        """Get the best downloaded model for a language."""
        for model_key in PREFERRED_MODELS.get(lang, []):
            if model_key in self._model_paths:
                return model_key
        return None

    # ─────────────────────────────────────────
    # Transcription
    # ─────────────────────────────────────────

    def transcribe(
        self,
        audio_input,
        language: str = "en",
        show_progress: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio in the specified language.
        
        Returns the same TranscriptionResult as the existing Transcriber,
        so it works with SmartAnalyzer, SmartDocumentGenerator, etc.
        
        Args:
            audio_input: File path (str/Path) or AudioData object
            language: 'en', 'hi', 'te', or 'auto'
            show_progress: Show progress messages
            
        Returns:
            TranscriptionResult (same type as Transcriber.transcribe)
        """
        if language.lower() == "auto":
            return self._transcribe_auto(audio_input, show_progress)

        lang_enum = LANGUAGE_MAP.get(language.lower())
        if not lang_enum:
            raise ValueError(f"Unsupported language: {language}. Use: en, hi, te, auto")

        model_key = self._get_best_model_for_language(lang_enum)
        if not model_key:
            raise FileNotFoundError(
                f"No model for '{language}'. Download with:\n"
                f"  python demo_speech.py --download-model {language}"
            )

        return self._transcribe_with_model(audio_input, model_key, lang_enum.value, show_progress)

    def _transcribe_with_model(
        self,
        audio_input,
        model_key: str,
        lang_code: str,
        show_progress: bool,
    ) -> TranscriptionResult:
        """Transcribe using a specific model. Returns standard TranscriptionResult."""
        # Load the model
        model = self._load_model(model_key)

        # Load audio (reuse Transcriber's _load_audio_file for format handling)
        temp_transcriber = Transcriber(sample_rate=self.SAMPLE_RATE)
        if isinstance(audio_input, (str, Path)):
            audio_data = temp_transcriber._load_audio_file(str(audio_input))
        else:
            audio_data = audio_input

        samples = audio_data.samples
        sr = audio_data.sample_rate

        # Preprocess (reuse Transcriber's preprocessing)
        samples = temp_transcriber._preprocess_audio(samples)

        # Resample if needed
        if sr != self.SAMPLE_RATE:
            samples = temp_transcriber._resample(samples, sr, self.SAMPLE_RATE)

        # Convert to PCM bytes
        audio_bytes = temp_transcriber._to_pcm_bytes(samples)

        # Create recognizer
        vosk = self._get_vosk()
        recognizer = vosk.KaldiRecognizer(model, self.SAMPLE_RATE)
        recognizer.SetWords(True)

        # Process in chunks
        all_words: List[Word] = []
        utterances: List[Utterance] = []
        chunk_size = 4000
        total_len = len(audio_bytes)

        if show_progress:
            model_label = MODEL_REGISTRY[model_key]["label"]
            print(f"[MultilingualTranscriber] Processing {audio_data.duration:.1f}s "
                  f"with {model_label} ({model_key})...")

        for i in range(0, total_len, chunk_size):
            chunk = audio_bytes[i:i + chunk_size]

            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                words, utterance = self._parse_result(result)
                all_words.extend(words)
                if utterance:
                    utterances.append(utterance)

            if show_progress and i % (chunk_size * 50) == 0:
                progress = min(100, int(i / total_len * 100))
                print(f"[MultilingualTranscriber] Progress: {progress}%", end='\r')

        # Final result
        final = json.loads(recognizer.FinalResult())
        words, utterance = self._parse_result(final)
        all_words.extend(words)
        if utterance:
            utterances.append(utterance)

        if show_progress:
            print(f"[MultilingualTranscriber] Progress: 100% - Complete!")

        full_text = ' '.join(w.text for w in all_words)

        return TranscriptionResult(
            text=full_text,
            utterances=utterances,
            words=all_words,
            duration=audio_data.duration,
            language=lang_code,
            model_name=model_key,
        )

    def _transcribe_auto(
        self, audio_input, show_progress: bool
    ) -> TranscriptionResult:
        """Auto-detect language by trying all available models on a sample."""
        available = {}
        for lang in Language:
            model_key = self._get_best_model_for_language(lang)
            if model_key:
                available[lang] = model_key

        if not available:
            raise FileNotFoundError("No language models downloaded.")

        if len(available) == 1:
            lang, model_key = next(iter(available.items()))
            return self._transcribe_with_model(audio_input, model_key, lang.value, show_progress)

        # Load audio once for sampling
        temp_transcriber = Transcriber(sample_rate=self.SAMPLE_RATE)
        if isinstance(audio_input, (str, Path)):
            audio_data = temp_transcriber._load_audio_file(str(audio_input))
        else:
            audio_data = audio_input

        # Take first 15 seconds as sample
        sample_frames = min(len(audio_data.samples), self.SAMPLE_RATE * 15)
        sample = audio_data.samples[:sample_frames].astype(np.float32)
        sample = np.clip(sample, -1.0, 1.0)
        sample_bytes = (sample * 32767).astype(np.int16).tobytes()

        if show_progress:
            print(f"[MultilingualTranscriber] Auto-detecting language "
                  f"({', '.join(l.value for l in available)})...")

        best_lang = None
        best_score = -1.0
        vosk = self._get_vosk()

        for lang, model_key in available.items():
            try:
                model = self._load_model(model_key)
                rec = vosk.KaldiRecognizer(model, self.SAMPLE_RATE)
                rec.SetWords(True)

                rec.AcceptWaveform(sample_bytes)
                result = json.loads(rec.FinalResult())

                text = result.get("text", "").strip()
                result_words = result.get("result", [])

                if result_words:
                    avg_conf = sum(w.get("conf", 0) for w in result_words) / len(result_words)
                else:
                    avg_conf = 0.0

                # Score = confidence * 0.6 + word count factor * 0.4
                score = avg_conf * 0.6 + min(len(text.split()), 50) / 50 * 0.4

                if show_progress:
                    print(f"   {lang.value} ({model_key}): "
                          f"conf={avg_conf:.3f}, words={len(text.split())}, score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_lang = lang

            except Exception as e:
                logger.warning(f"Error testing {lang.value}: {e}")

        if best_lang is None:
            best_lang = Language.ENGLISH

        if show_progress:
            print(f"   ✅ Detected: {best_lang.value}")

        model_key = available[best_lang]
        return self._transcribe_with_model(audio_input, model_key, best_lang.value, show_progress)

    def _parse_result(self, result: dict) -> Tuple[List[Word], Optional[Utterance]]:
        """Parse Vosk result into Word and Utterance objects (same as Transcriber)."""
        words = []

        if 'result' in result:
            for w in result['result']:
                word = Word(
                    text=w.get('word', ''),
                    start_time=w.get('start', 0.0),
                    end_time=w.get('end', 0.0),
                    confidence=w.get('conf', 1.0),
                )
                words.append(word)

        utterance = None
        if words:
            text = result.get('text', ' '.join(w.text for w in words))
            utterance = Utterance(
                text=text.strip(),
                words=words,
                start_time=words[0].start_time,
                end_time=words[-1].end_time,
            )

        return words, utterance
