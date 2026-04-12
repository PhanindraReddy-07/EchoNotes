"""
Whisper Transcriber for EchoNotes
===================================
Drop-in replacement for Vosk using OpenAI Whisper.

Returns the same TranscriptionResult as the existing Transcriber,
so it plugs directly into the NLP and document generation pipeline
without any other changes.

Installation:
    pip install openai-whisper

Model sizes:
    tiny   ~75MB  - fastest, lower accuracy
    base   ~145MB - good balance (recommended for most use)
    small  ~466MB - better accuracy (recommended for technical content)
    medium ~1.5GB - very good, slow on CPU
    large  ~3GB   - best, very slow on CPU

Usage:
    from speech.whisper_transcriber import WhisperTranscriber

    wt = WhisperTranscriber(model_size="small")
    result = wt.transcribe("lecture.wav", language="en")
    print(result.text)
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List

from .transcriber import Word, Utterance, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """
    Offline Speech-to-Text using OpenAI Whisper.

    Returns TranscriptionResult — same as Transcriber and MultilingualTranscriber —
    so it integrates with SmartAnalyzer and SmartDocumentGenerator without changes.

    Key advantages over Vosk:
    - Much better accuracy on Indian English, Hindi, Telugu
    - Handles technical vocabulary correctly
    - Handles background music / noise better
    - Automatic language detection

    Args:
        model_size: 'tiny', 'base', 'small', 'medium', 'large'
                    Recommended: 'small' for technical content
        device:     'cpu' or 'cuda' (auto-detected if None)
        language:   Default language hint ('en', 'hi', 'te', None for auto)
    """

    # Map EchoNotes language codes to Whisper language codes
    LANGUAGE_MAP = {
        'en': 'en',
        'en-in': 'en',
        'english': 'en',
        'hi': 'hi',
        'hindi': 'hi',
        'te': 'te',
        'telugu': 'te',
        'auto': None,
    }

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language = language
        self._model = None

        # Auto-detect device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def _load_model(self):
        """Lazy-load Whisper model."""
        if self._model is not None:
            return self._model

        try:
            import whisper
        except ImportError:
            raise ImportError(
                "Whisper is required:\n"
                "  pip install openai-whisper\n"
                "Also install ffmpeg: https://ffmpeg.org/download.html"
            )

        print(f"[WhisperTranscriber] Loading whisper-{self.model_size} on {self.device}...")
        self._model = whisper.load_model(self.model_size, device=self.device)
        print(f"[WhisperTranscriber] Model ready")
        return self._model

    def transcribe(
        self,
        audio_input,
        language: str = "en",
        show_progress: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio file or AudioData object.

        Args:
            audio_input: File path (str/Path) or AudioData object
            language:    Language code: 'en', 'hi', 'te', 'auto'
            show_progress: Print progress messages

        Returns:
            TranscriptionResult — same type as Vosk Transcriber
        """
        model = self._load_model()

        # Resolve language
        whisper_lang = self.LANGUAGE_MAP.get(language.lower(), 'en') if language else None

        # Get audio path or convert AudioData to numpy array
        audio_array, duration, file_path = self._prepare_audio(audio_input)

        if show_progress:
            src = file_path if file_path else f"AudioData ({duration:.1f}s)"
            print(f"[WhisperTranscriber] Transcribing: {src}")
            if whisper_lang:
                print(f"[WhisperTranscriber] Language: {whisper_lang}")
            else:
                print(f"[WhisperTranscriber] Language: auto-detect")

        # Run Whisper transcription
        transcribe_kwargs = {
            "verbose": False,
            "word_timestamps": True,   # enables word-level timing
            "task": "transcribe",
        }
        if whisper_lang:
            transcribe_kwargs["language"] = whisper_lang

        # Whisper accepts file path or numpy float32 array at 16kHz
        audio_input_for_whisper = file_path if file_path else audio_array
        result = model.transcribe(audio_input_for_whisper, **transcribe_kwargs)

        if show_progress:
            detected = result.get("language", whisper_lang or "unknown")
            print(f"[WhisperTranscriber] Detected language: {detected}")
            print(f"[WhisperTranscriber] Transcription complete")

        # Convert Whisper output to TranscriptionResult
        return self._build_result(result, duration, language or "en")

    def _prepare_audio(self, audio_input):
        """
        Prepare audio for Whisper.
        Returns (numpy_array, duration, file_path_or_None)
        """
        if isinstance(audio_input, (str, Path)):
            # File path — Whisper can handle it directly
            path = str(audio_input)
            # Get duration via pydub or wave
            duration = self._get_duration(path)
            return None, duration, path

        # AudioData object (from existing pipeline)
        samples = audio_input.samples.astype(np.float32)
        sr = audio_input.sample_rate
        duration = audio_input.duration

        # Resample to 16kHz if needed (Whisper requires 16kHz)
        if sr != 16000:
            samples = self._resample(samples, sr, 16000)

        # Whisper needs float32 in range [-1, 1]
        samples = np.clip(samples, -1.0, 1.0)

        return samples, duration, None

    def _get_duration(self, file_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import wave
            if file_path.endswith('.wav'):
                with wave.open(file_path, 'rb') as wf:
                    return wf.getnframes() / wf.getframerate()
        except Exception:
            pass
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0
        except Exception:
            pass
        return 0.0

    def _resample(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio."""
        try:
            from scipy import signal
            num_samples = int(len(samples) * target_sr / orig_sr)
            return signal.resample(samples, num_samples).astype(np.float32)
        except ImportError:
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(samples), 1 / ratio)
            indices = indices[indices < len(samples) - 1].astype(int)
            return samples[indices].astype(np.float32)

    def _build_result(self, whisper_result: dict, duration: float, language: str) -> TranscriptionResult:
        """
        Convert Whisper output dict to TranscriptionResult.

        Whisper result structure:
        {
            "text": "full transcript",
            "segments": [
                {
                    "text": "segment text",
                    "start": 0.0, "end": 5.2,
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.99},
                        ...
                    ]
                },
                ...
            ],
            "language": "en"
        }
        """
        all_words: List[Word] = []
        utterances: List[Utterance] = []

        segments = whisper_result.get("segments", [])

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)

            # Extract word-level data
            seg_words: List[Word] = []
            raw_words = seg.get("words", [])

            if raw_words:
                for w in raw_words:
                    word_text = w.get("word", "").strip()
                    if not word_text:
                        continue
                    word = Word(
                        text=word_text,
                        start_time=w.get("start", seg_start),
                        end_time=w.get("end", seg_end),
                        confidence=w.get("probability", 1.0),
                    )
                    seg_words.append(word)
                    all_words.append(word)
            else:
                # No word timestamps — create one word per segment
                word = Word(
                    text=seg_text,
                    start_time=seg_start,
                    end_time=seg_end,
                    confidence=1.0,
                )
                seg_words.append(word)
                all_words.append(word)

            if seg_words and seg_text:
                utterance = Utterance(
                    text=seg_text,
                    words=seg_words,
                    start_time=seg_start,
                    end_time=seg_end,
                )
                utterances.append(utterance)

        full_text = whisper_result.get("text", "").strip()
        if not full_text:
            full_text = " ".join(w.text for w in all_words)

        detected_lang = whisper_result.get("language", language)

        return TranscriptionResult(
            text=full_text,
            utterances=utterances,
            words=all_words,
            duration=duration,
            language=detected_lang,
            model_name=f"whisper-{self.model_size}",
        )

    def get_model_info(self) -> dict:
        """Return info about the loaded model."""
        return {
            "model": f"whisper-{self.model_size}",
            "device": self.device,
            "loaded": self._model is not None,
        }
