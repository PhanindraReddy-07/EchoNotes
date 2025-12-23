"""
LongFormProcessor - Process hours-long audio files
===================================================
Handles chunking, parallel processing, and result merging
"""
import numpy as np
from typing import List, Optional, Dict, Generator, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


@dataclass
class ProcessingProgress:
    """Progress information during processing"""
    current_chunk: int
    total_chunks: int
    elapsed_time: float
    estimated_remaining: float
    current_phase: str  # 'transcribing', 'diarizing', 'merging'
    
    @property
    def progress_percent(self) -> float:
        return (self.current_chunk / self.total_chunks) * 100
    
    def __str__(self) -> str:
        return (
            f"[{self.current_phase}] Chunk {self.current_chunk}/{self.total_chunks} "
            f"({self.progress_percent:.1f}%) - "
            f"Elapsed: {self.elapsed_time:.1f}s, "
            f"Remaining: {self.estimated_remaining:.1f}s"
        )


class LongFormProcessor:
    """
    Process long audio files efficiently
    
    Features:
    - Automatic chunking with overlap
    - Progress tracking
    - Memory-efficient processing
    - Result merging with deduplication
    - Optional parallel processing
    
    Usage:
        processor = LongFormProcessor(
            transcriber=transcriber,
            diarizer=diarizer,
            chunk_duration=60.0
        )
        
        result = processor.process(audio_data, progress_callback=print)
    """
    
    def __init__(
        self,
        transcriber,  # Transcriber instance
        diarizer=None,  # Optional SpeakerDiarizer instance
        chunk_duration: float = 60.0,
        chunk_overlap: float = 2.0,
        max_workers: int = 1  # Set > 1 for parallel processing
    ):
        """
        Initialize the long-form processor
        
        Args:
            transcriber: Transcriber instance for ASR
            diarizer: Optional SpeakerDiarizer for speaker identification
            chunk_duration: Duration of each chunk in seconds
            chunk_overlap: Overlap between chunks in seconds
            max_workers: Number of parallel workers (1 = sequential)
        """
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.chunk_duration = chunk_duration
        self.chunk_overlap = chunk_overlap
        self.max_workers = max_workers
    
    def process(
        self,
        audio_data,  # AudioData from audio module
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        include_diarization: bool = True
    ):
        """
        Process long audio file
        
        Args:
            audio_data: AudioData object
            progress_callback: Optional callback for progress updates
            include_diarization: Whether to perform speaker diarization
            
        Returns:
            TranscriptionResult with merged results
        """
        from .transcriber import TranscriptionResult, Utterance, Word
        
        start_time = time.time()
        
        # Split into chunks
        chunks = self._create_chunks(audio_data)
        total_chunks = len(chunks)
        
        if progress_callback:
            progress_callback(ProcessingProgress(
                current_chunk=0,
                total_chunks=total_chunks,
                elapsed_time=0,
                estimated_remaining=0,
                current_phase='starting'
            ))
        
        print(f"[LongFormProcessor] Processing {audio_data.duration:.1f}s audio in {total_chunks} chunks")
        
        # Process chunks
        all_results = []
        
        if self.max_workers > 1 and total_chunks > 1:
            # Parallel processing
            all_results = self._process_parallel(chunks, progress_callback, start_time)
        else:
            # Sequential processing
            all_results = self._process_sequential(chunks, progress_callback, start_time)
        
        # Merge results
        if progress_callback:
            elapsed = time.time() - start_time
            progress_callback(ProcessingProgress(
                current_chunk=total_chunks,
                total_chunks=total_chunks,
                elapsed_time=elapsed,
                estimated_remaining=0,
                current_phase='merging'
            ))
        
        merged_result = self._merge_results(all_results, audio_data.duration)
        
        # Optional diarization
        if include_diarization and self.diarizer:
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback(ProcessingProgress(
                    current_chunk=total_chunks,
                    total_chunks=total_chunks,
                    elapsed_time=elapsed,
                    estimated_remaining=0,
                    current_phase='diarizing'
                ))
            
            diarization = self.diarizer.diarize(audio_data, show_progress=False)
            self.diarizer.assign_speakers_to_transcript(diarization, merged_result)
        
        total_time = time.time() - start_time
        print(f"[LongFormProcessor] Complete in {total_time:.1f}s")
        
        return merged_result
    
    def process_streaming(
        self,
        audio_data,
        callback: Callable[[str], None]
    ) -> Generator[str, None, None]:
        """
        Process with streaming output (yields partial results)
        
        Args:
            audio_data: AudioData object
            callback: Callback for each transcribed chunk
            
        Yields:
            Partial transcription strings
        """
        chunks = self._create_chunks(audio_data)
        
        for i, (chunk, offset) in enumerate(chunks):
            print(f"[LongFormProcessor] Chunk {i+1}/{len(chunks)}")
            
            result = self.transcriber.transcribe(chunk, show_progress=False)
            
            # Adjust timestamps
            for word in result.words:
                word.start_time += offset
                word.end_time += offset
            
            callback(result.text)
            yield result.text
    
    def _create_chunks(self, audio_data) -> List[tuple]:
        """
        Split audio into overlapping chunks
        
        Returns:
            List of (AudioData chunk, time offset) tuples
        """
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from audio import AudioData
        
        samples = audio_data.samples
        sr = audio_data.sample_rate
        
        chunk_samples = int(self.chunk_duration * sr)
        overlap_samples = int(self.chunk_overlap * sr)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        
        for i in range(0, len(samples), step_samples):
            end = min(i + chunk_samples, len(samples))
            chunk_data = samples[i:end]
            
            # Skip very short final chunks
            if len(chunk_data) < sr * 1.0:
                break
            
            offset = i / sr
            
            chunk = AudioData(
                samples=chunk_data,
                sample_rate=sr,
                channels=audio_data.channels,
                duration=len(chunk_data) / sr,
                source=audio_data.source,
                filepath=audio_data.filepath
            )
            
            chunks.append((chunk, offset))
        
        return chunks
    
    def _process_sequential(
        self,
        chunks: List[tuple],
        progress_callback: Optional[Callable],
        start_time: float
    ) -> List[tuple]:
        """Process chunks sequentially"""
        results = []
        
        for i, (chunk, offset) in enumerate(chunks):
            result = self.transcriber.transcribe(chunk, show_progress=False)
            results.append((result, offset))
            
            if progress_callback:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(chunks) - i - 1)
                
                progress_callback(ProcessingProgress(
                    current_chunk=i + 1,
                    total_chunks=len(chunks),
                    elapsed_time=elapsed,
                    estimated_remaining=remaining,
                    current_phase='transcribing'
                ))
        
        return results
    
    def _process_parallel(
        self,
        chunks: List[tuple],
        progress_callback: Optional[Callable],
        start_time: float
    ) -> List[tuple]:
        """Process chunks in parallel"""
        results = [None] * len(chunks)
        completed = 0
        
        def process_chunk(args):
            idx, chunk, offset = args
            result = self.transcriber.transcribe(chunk, show_progress=False)
            return idx, result, offset
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_chunk, (i, chunk, offset)): i
                for i, (chunk, offset) in enumerate(chunks)
            }
            
            for future in as_completed(futures):
                idx, result, offset = future.result()
                results[idx] = (result, offset)
                completed += 1
                
                if progress_callback:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = avg_time * (len(chunks) - completed)
                    
                    progress_callback(ProcessingProgress(
                        current_chunk=completed,
                        total_chunks=len(chunks),
                        elapsed_time=elapsed,
                        estimated_remaining=remaining,
                        current_phase='transcribing'
                    ))
        
        return results
    
    def _merge_results(self, results: List[tuple], total_duration: float):
        """
        Merge chunk results, handling overlaps
        
        Deduplicates words in overlap regions using timestamps.
        """
        from .transcriber import TranscriptionResult, Utterance, Word
        
        if not results:
            return TranscriptionResult(
                text="",
                utterances=[],
                words=[],
                duration=total_duration,
                model_name=self.transcriber.model_name
            )
        
        all_words = []
        all_utterances = []
        
        for result, offset in results:
            # Adjust timestamps
            for word in result.words:
                adjusted_word = Word(
                    text=word.text,
                    start_time=word.start_time + offset,
                    end_time=word.end_time + offset,
                    confidence=word.confidence
                )
                all_words.append(adjusted_word)
            
            for utterance in result.utterances:
                adjusted_utterance = Utterance(
                    text=utterance.text,
                    words=[],  # Will be reconstructed
                    start_time=utterance.start_time + offset,
                    end_time=utterance.end_time + offset,
                    speaker=utterance.speaker
                )
                all_utterances.append(adjusted_utterance)
        
        # Sort by start time
        all_words.sort(key=lambda w: w.start_time)
        
        # Remove duplicates in overlap regions
        deduplicated_words = self._deduplicate_words(all_words)
        
        # Rebuild text
        full_text = ' '.join(w.text for w in deduplicated_words)
        
        return TranscriptionResult(
            text=full_text,
            utterances=all_utterances,
            words=deduplicated_words,
            duration=total_duration,
            model_name=self.transcriber.model_name
        )
    
    def _deduplicate_words(self, words: List, time_threshold: float = 0.1) -> List:
        """Remove duplicate words in overlap regions"""
        if not words:
            return words
        
        deduplicated = [words[0]]
        
        for word in words[1:]:
            prev = deduplicated[-1]
            
            # Check if this is a duplicate (same word, similar time)
            time_diff = abs(word.start_time - prev.start_time)
            
            if time_diff < time_threshold and word.text.lower() == prev.text.lower():
                # Keep the one with higher confidence
                if word.confidence > prev.confidence:
                    deduplicated[-1] = word
            else:
                deduplicated.append(word)
        
        return deduplicated
    
    def estimate_processing_time(self, audio_duration: float) -> Dict:
        """
        Estimate processing time for given audio duration
        
        Based on typical processing speeds.
        
        Args:
            audio_duration: Audio duration in seconds
            
        Returns:
            Dict with time estimates
        """
        # Typical speeds (audio seconds per processing second)
        # These are rough estimates, actual speed depends on hardware
        asr_speed = 3.0  # 3x realtime for ASR
        diarization_speed = 5.0  # 5x realtime for diarization
        
        num_chunks = int(np.ceil(audio_duration / (self.chunk_duration - self.chunk_overlap)))
        
        asr_time = audio_duration / asr_speed
        diarization_time = audio_duration / diarization_speed if self.diarizer else 0
        
        total_time = asr_time + diarization_time
        
        return {
            'audio_duration': round(audio_duration, 1),
            'num_chunks': num_chunks,
            'estimated_asr_time': round(asr_time, 1),
            'estimated_diarization_time': round(diarization_time, 1),
            'estimated_total_time': round(total_time, 1),
            'note': 'Estimates based on typical processing speeds'
        }
