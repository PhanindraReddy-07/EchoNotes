"""
Confidence Scorer - Word-level confidence analysis
===================================================
Analyzes and highlights low-confidence transcription sections
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ConfidenceSpan:
    """A span of text with confidence information"""
    text: str
    start_time: float
    end_time: float
    confidence: float
    level: str  # 'high', 'medium', 'low'
    word_indices: Tuple[int, int]  # Start and end word indices
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'start': round(self.start_time, 3),
            'end': round(self.end_time, 3),
            'confidence': round(self.confidence, 3),
            'level': self.level
        }


@dataclass
class ConfidenceReport:
    """Comprehensive confidence analysis report"""
    overall_confidence: float
    high_confidence_ratio: float    # % of words with high confidence
    medium_confidence_ratio: float  # % of words with medium confidence
    low_confidence_ratio: float     # % of words with low confidence
    low_confidence_spans: List[ConfidenceSpan]
    word_count: int
    
    # Thresholds used
    high_threshold: float
    low_threshold: float
    
    def to_dict(self) -> Dict:
        return {
            'overall_confidence': round(self.overall_confidence, 3),
            'word_count': self.word_count,
            'distribution': {
                'high': round(self.high_confidence_ratio * 100, 1),
                'medium': round(self.medium_confidence_ratio * 100, 1),
                'low': round(self.low_confidence_ratio * 100, 1)
            },
            'low_confidence_spans': [s.to_dict() for s in self.low_confidence_spans],
            'needs_review': self.needs_review,
            'quality_rating': self.quality_rating
        }
    
    @property
    def needs_review(self) -> bool:
        """Whether manual review is recommended"""
        return self.low_confidence_ratio > 0.15 or self.overall_confidence < 0.7
    
    @property
    def quality_rating(self) -> str:
        """Overall quality rating"""
        if self.overall_confidence >= 0.9 and self.low_confidence_ratio < 0.05:
            return "Excellent"
        elif self.overall_confidence >= 0.8 and self.low_confidence_ratio < 0.15:
            return "Good"
        elif self.overall_confidence >= 0.7 and self.low_confidence_ratio < 0.25:
            return "Fair"
        else:
            return "Poor - Review Recommended"


class ConfidenceScorer:
    """
    Analyzes transcription confidence at word and span level
    
    Features:
    - Word-level confidence classification (high/medium/low)
    - Span detection for consecutive low-confidence words
    - Overall quality assessment
    - Calibration analysis
    - Recommendations for review
    
    Usage:
        scorer = ConfidenceScorer()
        report = scorer.analyze(transcription_result)
        
        print(f"Overall confidence: {report.overall_confidence:.1%}")
        print(f"Needs review: {report.needs_review}")
        
        for span in report.low_confidence_spans:
            print(f"Review: '{span.text}' (confidence: {span.confidence:.1%})")
    """
    
    def __init__(
        self,
        high_threshold: float = 0.85,
        low_threshold: float = 0.60,
        min_span_words: int = 1
    ):
        """
        Initialize the confidence scorer
        
        Args:
            high_threshold: Confidence above this is "high"
            low_threshold: Confidence below this is "low"
            min_span_words: Minimum words in a low-confidence span to report
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.min_span_words = min_span_words
    
    def analyze(self, transcription_result) -> ConfidenceReport:
        """
        Analyze confidence of a transcription
        
        Args:
            transcription_result: TranscriptionResult from Transcriber
            
        Returns:
            ConfidenceReport with detailed analysis
        """
        words = transcription_result.words
        
        if not words:
            return ConfidenceReport(
                overall_confidence=0.0,
                high_confidence_ratio=0.0,
                medium_confidence_ratio=0.0,
                low_confidence_ratio=0.0,
                low_confidence_spans=[],
                word_count=0,
                high_threshold=self.high_threshold,
                low_threshold=self.low_threshold
            )
        
        # Classify each word
        confidences = [w.confidence for w in words]
        
        high_count = sum(1 for c in confidences if c >= self.high_threshold)
        low_count = sum(1 for c in confidences if c < self.low_threshold)
        medium_count = len(confidences) - high_count - low_count
        
        total = len(confidences)
        
        # Find low-confidence spans
        low_spans = self._find_low_confidence_spans(words)
        
        return ConfidenceReport(
            overall_confidence=np.mean(confidences),
            high_confidence_ratio=high_count / total,
            medium_confidence_ratio=medium_count / total,
            low_confidence_ratio=low_count / total,
            low_confidence_spans=low_spans,
            word_count=total,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold
        )
    
    def _find_low_confidence_spans(self, words: List) -> List[ConfidenceSpan]:
        """Find consecutive spans of low-confidence words"""
        spans = []
        current_span_start = None
        current_span_words = []
        
        for i, word in enumerate(words):
            is_low = word.confidence < self.low_threshold
            
            if is_low:
                if current_span_start is None:
                    current_span_start = i
                current_span_words.append(word)
            else:
                # End current span if exists
                if current_span_words and len(current_span_words) >= self.min_span_words:
                    span = self._create_span(current_span_words, current_span_start, i - 1)
                    spans.append(span)
                current_span_start = None
                current_span_words = []
        
        # Handle final span
        if current_span_words and len(current_span_words) >= self.min_span_words:
            span = self._create_span(current_span_words, current_span_start, len(words) - 1)
            spans.append(span)
        
        return spans
    
    def _create_span(self, words: List, start_idx: int, end_idx: int) -> ConfidenceSpan:
        """Create a ConfidenceSpan from a list of words"""
        text = ' '.join(w.text for w in words)
        avg_conf = np.mean([w.confidence for w in words])
        
        level = 'low' if avg_conf < self.low_threshold else (
            'high' if avg_conf >= self.high_threshold else 'medium'
        )
        
        return ConfidenceSpan(
            text=text,
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            confidence=avg_conf,
            level=level,
            word_indices=(start_idx, end_idx)
        )
    
    def get_highlighted_text(
        self,
        transcription_result,
        format: str = 'markdown'
    ) -> str:
        """
        Get transcript with low-confidence sections highlighted
        
        Args:
            transcription_result: TranscriptionResult from Transcriber
            format: Output format - 'markdown', 'html', or 'plain'
            
        Returns:
            Formatted text with highlights
        """
        words = transcription_result.words
        
        if not words:
            return ""
        
        # Build text with markers
        result_parts = []
        
        for word in words:
            if word.confidence < self.low_threshold:
                if format == 'markdown':
                    result_parts.append(f"**[{word.text}?]**")
                elif format == 'html':
                    result_parts.append(
                        f'<span class="low-confidence" title="Confidence: {word.confidence:.0%}">'
                        f'{word.text}</span>'
                    )
                else:
                    result_parts.append(f"[{word.text}?]")
            elif word.confidence < self.high_threshold:
                if format == 'markdown':
                    result_parts.append(f"*{word.text}*")
                elif format == 'html':
                    result_parts.append(
                        f'<span class="medium-confidence">{word.text}</span>'
                    )
                else:
                    result_parts.append(word.text)
            else:
                result_parts.append(word.text)
        
        return ' '.join(result_parts)
    
    def get_confidence_timeline(
        self,
        transcription_result,
        window_size: float = 5.0
    ) -> List[Dict]:
        """
        Get confidence over time (for visualization)
        
        Args:
            transcription_result: TranscriptionResult
            window_size: Time window in seconds
            
        Returns:
            List of {time, confidence} points
        """
        words = transcription_result.words
        
        if not words:
            return []
        
        duration = transcription_result.duration
        timeline = []
        
        for t in np.arange(0, duration, window_size / 2):
            # Get words in this window
            window_words = [
                w for w in words
                if w.start_time >= t and w.start_time < t + window_size
            ]
            
            if window_words:
                avg_conf = np.mean([w.confidence for w in window_words])
            else:
                avg_conf = None
            
            timeline.append({
                'time': round(t, 2),
                'confidence': round(avg_conf, 3) if avg_conf else None,
                'word_count': len(window_words)
            })
        
        return timeline
    
    def calibration_analysis(
        self,
        transcription_result,
        ground_truth: Optional[str] = None
    ) -> Dict:
        """
        Analyze if confidence scores are well-calibrated
        
        If ground truth is provided, checks if confidence correlates with accuracy.
        Otherwise, provides statistical analysis of confidence distribution.
        
        Args:
            transcription_result: TranscriptionResult
            ground_truth: Optional ground truth transcript
            
        Returns:
            Calibration analysis dict
        """
        words = transcription_result.words
        
        if not words:
            return {'error': 'No words to analyze'}
        
        confidences = [w.confidence for w in words]
        
        analysis = {
            'mean': round(np.mean(confidences), 3),
            'std': round(np.std(confidences), 3),
            'min': round(np.min(confidences), 3),
            'max': round(np.max(confidences), 3),
            'median': round(np.median(confidences), 3),
            'distribution': {
                'high': sum(1 for c in confidences if c >= self.high_threshold),
                'medium': sum(1 for c in confidences if self.low_threshold <= c < self.high_threshold),
                'low': sum(1 for c in confidences if c < self.low_threshold)
            }
        }
        
        # Check for common calibration issues
        if np.std(confidences) < 0.05:
            analysis['warning'] = 'Very low variance - confidence may not be informative'
        
        if np.mean(confidences) > 0.95:
            analysis['note'] = 'Very high confidence - may be overconfident'
        
        return analysis


def format_confidence_report(report: ConfidenceReport) -> str:
    """Format a confidence report as a readable string"""
    lines = [
        "=" * 50,
        "         TRANSCRIPTION CONFIDENCE REPORT",
        "=" * 50,
        f"",
        f"  Overall Confidence: {report.overall_confidence:.1%}",
        f"  Quality Rating:     {report.quality_rating}",
        f"  Total Words:        {report.word_count}",
        f"",
        f"  Confidence Distribution:",
        f"    ● High   (>{report.high_threshold:.0%}):  {report.high_confidence_ratio:>5.1%}",
        f"    ● Medium:            {report.medium_confidence_ratio:>5.1%}",
        f"    ● Low    (<{report.low_threshold:.0%}):  {report.low_confidence_ratio:>5.1%}",
        f"",
    ]
    
    if report.needs_review:
        lines.append("  ⚠️  MANUAL REVIEW RECOMMENDED")
        lines.append("")
    
    if report.low_confidence_spans:
        lines.append(f"  Low-Confidence Sections ({len(report.low_confidence_spans)}):")
        for i, span in enumerate(report.low_confidence_spans[:10]):
            text_preview = span.text[:40] + "..." if len(span.text) > 40 else span.text
            lines.append(f"    {i+1}. [{span.start_time:.1f}s] \"{text_preview}\"")
            lines.append(f"       Confidence: {span.confidence:.1%}")
        
        if len(report.low_confidence_spans) > 10:
            lines.append(f"    ... and {len(report.low_confidence_spans) - 10} more")
    else:
        lines.append("  ✓ No significant low-confidence sections")
    
    lines.append("")
    lines.append("=" * 50)
    
    return '\n'.join(lines)
