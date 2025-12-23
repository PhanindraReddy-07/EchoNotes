"""
Speaker Diarizer - Who spoke when?
==================================
Identifies different speakers in audio using voice embeddings
"""
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

# Lazy imports
def _get_resemblyzer():
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        return VoiceEncoder, preprocess_wav
    except ImportError:
        raise ImportError(
            "Resemblyzer required for speaker diarization.\n"
            "Install with: pip install resemblyzer"
        )


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a specific speaker"""
    speaker_id: str          # "Speaker 1", "Speaker 2", etc.
    start_time: float        # seconds
    end_time: float          # seconds
    confidence: float = 1.0  # Confidence in speaker assignment
    embedding: Optional[np.ndarray] = None  # Voice embedding (optional)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            'speaker': self.speaker_id,
            'start': round(self.start_time, 3),
            'end': round(self.end_time, 3),
            'duration': round(self.duration, 3),
            'confidence': round(self.confidence, 3)
        }


@dataclass
class DiarizationResult:
    """Complete diarization result"""
    segments: List[SpeakerSegment]
    num_speakers: int
    speaker_stats: Dict[str, Dict]  # Stats per speaker
    duration: float
    
    def get_speaker_segments(self, speaker_id: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker"""
        return [s for s in self.segments if s.speaker_id == speaker_id]
    
    def get_speaker_at_time(self, time: float) -> Optional[str]:
        """Get speaker ID at a specific time"""
        for seg in self.segments:
            if seg.start_time <= time <= seg.end_time:
                return seg.speaker_id
        return None
    
    def to_dict(self) -> Dict:
        return {
            'num_speakers': self.num_speakers,
            'duration': round(self.duration, 2),
            'speaker_stats': self.speaker_stats,
            'segments': [s.to_dict() for s in self.segments]
        }
    
    def get_timeline(self) -> str:
        """Get a visual timeline of speakers"""
        lines = []
        for seg in self.segments:
            time_str = f"{seg.start_time:>6.1f}s - {seg.end_time:>6.1f}s"
            bar_len = min(30, int(seg.duration * 3))
            bar = "█" * bar_len
            lines.append(f"{seg.speaker_id:<12} [{time_str}] {bar}")
        return '\n'.join(lines)


class SpeakerDiarizer:
    """
    Speaker Diarization using Voice Embeddings
    
    Identifies WHO spoke WHEN in an audio recording.
    Uses Resemblyzer for voice embeddings and clustering.
    
    Features:
    - Automatic speaker count detection
    - Voice embedding extraction
    - Clustering for speaker identification
    - Speaker merging across segments
    
    Usage:
        diarizer = SpeakerDiarizer()
        result = diarizer.diarize(audio_data)
        
        for segment in result.segments:
            print(f"{segment.speaker_id}: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
    """
    
    def __init__(
        self,
        min_speakers: int = 1,
        max_speakers: int = 10,
        segment_duration: float = 1.5,  # seconds per segment for embedding
        min_segment_duration: float = 0.5,
        use_gpu: bool = False
    ):
        """
        Initialize the diarizer
        
        Args:
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers
            segment_duration: Duration of segments for embedding extraction
            min_segment_duration: Minimum segment duration to consider
            use_gpu: Use GPU for embedding extraction
        """
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.segment_duration = segment_duration
        self.min_segment_duration = min_segment_duration
        self.use_gpu = use_gpu
        
        self._encoder = None
        self._preprocess_wav = None
    
    def _load_encoder(self):
        """Lazy load the voice encoder"""
        if self._encoder is None:
            VoiceEncoder, preprocess_wav = _get_resemblyzer()
            device = "cuda" if self.use_gpu else "cpu"
            self._encoder = VoiceEncoder(device=device)
            self._preprocess_wav = preprocess_wav
            print("[Diarizer] Voice encoder loaded")
    
    def diarize(
        self,
        audio_data,  # AudioData from audio module
        num_speakers: Optional[int] = None,
        show_progress: bool = True
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio
        
        Args:
            audio_data: AudioData object from audio module
            num_speakers: Number of speakers (None for auto-detect)
            show_progress: Show progress during processing
            
        Returns:
            DiarizationResult with speaker segments
        """
        self._load_encoder()
        
        samples = audio_data.samples
        sr = audio_data.sample_rate
        duration = audio_data.duration
        
        if show_progress:
            print(f"[Diarizer] Processing {duration:.1f}s of audio...")
        
        # Preprocess audio for Resemblyzer (expects specific format)
        if sr != 16000:
            samples = self._resample(samples, sr, 16000)
            sr = 16000
        
        # Normalize
        samples = samples / (np.max(np.abs(samples)) + 1e-10)
        
        # Extract embeddings for sliding windows
        embeddings = []
        timestamps = []
        
        window_samples = int(self.segment_duration * sr)
        hop_samples = window_samples // 2  # 50% overlap
        
        if show_progress:
            print("[Diarizer] Extracting voice embeddings...")
        
        for i in range(0, len(samples) - window_samples, hop_samples):
            segment = samples[i:i + window_samples]
            
            # Check if segment has enough energy (voice activity)
            energy = np.sum(segment ** 2) / len(segment)
            if energy < 0.001:  # Skip silent segments
                continue
            
            try:
                embedding = self._encoder.embed_utterance(segment)
                embeddings.append(embedding)
                timestamps.append(i / sr)
            except Exception:
                continue
        
        if not embeddings:
            print("[Diarizer] No speech detected for diarization")
            return DiarizationResult(
                segments=[],
                num_speakers=0,
                speaker_stats={},
                duration=duration
            )
        
        embeddings = np.array(embeddings)
        
        if show_progress:
            print(f"[Diarizer] Extracted {len(embeddings)} embeddings")
        
        # Cluster embeddings to find speakers
        if num_speakers is None:
            num_speakers = self._estimate_num_speakers(embeddings)
        
        num_speakers = max(self.min_speakers, min(self.max_speakers, num_speakers))
        
        if show_progress:
            print(f"[Diarizer] Clustering into {num_speakers} speakers...")
        
        labels = self._cluster_embeddings(embeddings, num_speakers)
        
        # Build speaker segments
        segments = self._build_segments(timestamps, labels, duration)
        
        # Merge adjacent segments from same speaker
        segments = self._merge_adjacent_segments(segments)
        
        # Calculate speaker statistics
        speaker_stats = self._calculate_stats(segments, duration)
        
        if show_progress:
            print(f"[Diarizer] Found {num_speakers} speakers in {len(segments)} segments")
        
        return DiarizationResult(
            segments=segments,
            num_speakers=num_speakers,
            speaker_stats=speaker_stats,
            duration=duration
        )
    
    def _estimate_num_speakers(self, embeddings: np.ndarray) -> int:
        """Estimate number of speakers using eigenvalue analysis"""
        try:
            from sklearn.cluster import SpectralClustering
            from scipy.spatial.distance import cdist
            
            # Calculate similarity matrix
            similarity = 1 - cdist(embeddings, embeddings, metric='cosine')
            
            # Eigenvalue analysis
            eigenvalues = np.linalg.eigvalsh(similarity)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Find elbow in eigenvalue curve
            diffs = np.diff(eigenvalues[:self.max_speakers])
            if len(diffs) > 0:
                num_speakers = np.argmax(np.abs(diffs)) + 1
            else:
                num_speakers = 2
            
            return max(self.min_speakers, min(self.max_speakers, num_speakers))
            
        except ImportError:
            # Fallback: assume 2 speakers
            return 2
    
    def _cluster_embeddings(self, embeddings: np.ndarray, num_speakers: int) -> np.ndarray:
        """Cluster embeddings to assign speaker labels"""
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            return labels
            
        except ImportError:
            # Fallback: simple k-means style clustering
            return self._simple_cluster(embeddings, num_speakers)
    
    def _simple_cluster(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        """Simple clustering fallback when sklearn not available"""
        n = len(embeddings)
        
        # Initialize centroids randomly
        indices = np.random.choice(n, k, replace=False)
        centroids = embeddings[indices].copy()
        
        labels = np.zeros(n, dtype=int)
        
        for _ in range(20):  # Max iterations
            # Assign labels
            for i, emb in enumerate(embeddings):
                distances = [np.linalg.norm(emb - c) for c in centroids]
                labels[i] = np.argmin(distances)
            
            # Update centroids
            new_centroids = []
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids.append(np.mean(embeddings[mask], axis=0))
                else:
                    new_centroids.append(centroids[j])
            centroids = np.array(new_centroids)
        
        return labels
    
    def _build_segments(
        self,
        timestamps: List[float],
        labels: np.ndarray,
        total_duration: float
    ) -> List[SpeakerSegment]:
        """Build speaker segments from timestamps and labels"""
        segments = []
        
        for i, (time, label) in enumerate(zip(timestamps, labels)):
            # Determine segment end time
            if i < len(timestamps) - 1:
                end_time = timestamps[i + 1]
            else:
                end_time = min(time + self.segment_duration, total_duration)
            
            segments.append(SpeakerSegment(
                speaker_id=f"Speaker {label + 1}",
                start_time=time,
                end_time=end_time
            ))
        
        return segments
    
    def _merge_adjacent_segments(
        self,
        segments: List[SpeakerSegment],
        max_gap: float = 0.5
    ) -> List[SpeakerSegment]:
        """Merge adjacent segments from the same speaker"""
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for seg in segments[1:]:
            prev = merged[-1]
            gap = seg.start_time - prev.end_time
            
            if prev.speaker_id == seg.speaker_id and gap <= max_gap:
                # Merge with previous
                merged[-1] = SpeakerSegment(
                    speaker_id=prev.speaker_id,
                    start_time=prev.start_time,
                    end_time=seg.end_time
                )
            else:
                merged.append(seg)
        
        # Filter out very short segments
        merged = [s for s in merged if s.duration >= self.min_segment_duration]
        
        return merged
    
    def _calculate_stats(
        self,
        segments: List[SpeakerSegment],
        total_duration: float
    ) -> Dict[str, Dict]:
        """Calculate statistics for each speaker"""
        stats = {}
        
        for seg in segments:
            speaker = seg.speaker_id
            if speaker not in stats:
                stats[speaker] = {
                    'total_time': 0.0,
                    'num_segments': 0,
                    'percentage': 0.0
                }
            
            stats[speaker]['total_time'] += seg.duration
            stats[speaker]['num_segments'] += 1
        
        # Calculate percentages
        for speaker in stats:
            stats[speaker]['percentage'] = (
                stats[speaker]['total_time'] / total_duration * 100
            )
            stats[speaker]['total_time'] = round(stats[speaker]['total_time'], 2)
            stats[speaker]['percentage'] = round(stats[speaker]['percentage'], 1)
        
        return stats
    
    def _resample(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio"""
        try:
            from scipy import signal
            num_samples = int(len(samples) * target_sr / orig_sr)
            return signal.resample(samples, num_samples)
        except ImportError:
            ratio = target_sr / orig_sr
            indices = np.arange(0, len(samples), 1/ratio)
            indices = indices[indices < len(samples) - 1].astype(int)
            return samples[indices]
    
    def assign_speakers_to_transcript(
        self,
        diarization_result: DiarizationResult,
        transcription_result  # TranscriptionResult from transcriber
    ) -> None:
        """
        Assign speaker IDs to transcription utterances
        
        Modifies transcription_result in place, adding speaker info.
        
        Args:
            diarization_result: Result from diarize()
            transcription_result: Result from Transcriber.transcribe()
        """
        for utterance in transcription_result.utterances:
            # Find speaker at utterance midpoint
            mid_time = (utterance.start_time + utterance.end_time) / 2
            speaker = diarization_result.get_speaker_at_time(mid_time)
            
            if speaker is None:
                # Try start time
                speaker = diarization_result.get_speaker_at_time(utterance.start_time)
            
            utterance.speaker = speaker or "Unknown"
