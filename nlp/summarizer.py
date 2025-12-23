"""
Hybrid Summarization Engine
============================
Combines multiple summarization strategies for better results:
1. Transformer-based abstractive (BART/T5)
2. Extractive clustering (sentence embeddings)
3. TextRank extractive
4. Meeting-aware pattern extraction

TECHNICAL NOVELTY:
- Ensemble approach reduces hallucination
- Adaptive strategy selection based on content type
- Weighted combination based on confidence scores
"""
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SummaryResult:
    """Result from summarization"""
    summary: str
    strategy_used: str
    confidence: float
    key_points: List[str]
    word_count: int
    compression_ratio: float
    
    def to_dict(self) -> Dict:
        return {
            'summary': self.summary,
            'strategy': self.strategy_used,
            'confidence': round(self.confidence, 3),
            'key_points': self.key_points,
            'word_count': self.word_count,
            'compression_ratio': round(self.compression_ratio, 2)
        }


class TextRankSummarizer:
    """
    TextRank-based extractive summarization
    
    Uses graph-based ranking to identify important sentences.
    No external dependencies required.
    """
    
    def __init__(self, damping: float = 0.85, min_similarity: float = 0.1):
        self.damping = damping
        self.min_similarity = min_similarity
    
    def summarize(
        self,
        text: str,
        num_sentences: int = 3,
        min_sentence_length: int = 10
    ) -> Tuple[str, List[str]]:
        """
        Extract key sentences using TextRank
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            min_sentence_length: Minimum characters per sentence
            
        Returns:
            Tuple of (summary string, list of key sentences)
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        sentences = [s for s in sentences if len(s) >= min_sentence_length]
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences), sentences
        
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Apply TextRank
        scores = self._textrank(similarity_matrix)
        
        # Get top sentences (maintain original order)
        ranked_indices = np.argsort(scores)[::-1][:num_sentences]
        ranked_indices = sorted(ranked_indices)  # Maintain order
        
        key_sentences = [sentences[i] for i in ranked_indices]
        summary = ' '.join(key_sentences)
        
        return summary, key_sentences
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Build sentence similarity matrix using word overlap"""
        n = len(sentences)
        matrix = np.zeros((n, n))
        
        # Tokenize sentences
        tokenized = [set(self._tokenize(s)) for s in sentences]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Jaccard similarity
                    intersection = len(tokenized[i] & tokenized[j])
                    union = len(tokenized[i] | tokenized[j])
                    if union > 0:
                        similarity = intersection / union
                        if similarity >= self.min_similarity:
                            matrix[i][j] = similarity
        
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums
        
        return matrix
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
                    'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her'}
        return [w for w in words if w not in stopwords]
    
    def _textrank(self, matrix: np.ndarray, iterations: int = 100) -> np.ndarray:
        """Apply TextRank algorithm"""
        n = matrix.shape[0]
        scores = np.ones(n) / n
        
        for _ in range(iterations):
            new_scores = (1 - self.damping) / n + self.damping * matrix.T @ scores
            if np.allclose(scores, new_scores, atol=1e-6):
                break
            scores = new_scores
        
        return scores


class ClusteringSummarizer:
    """
    Clustering-based extractive summarization
    
    Groups similar sentences and picks representatives.
    Uses sentence embeddings when available.
    """
    
    def __init__(self):
        self.embedder = None
    
    def _load_embedder(self):
        """Load sentence transformer model"""
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                print("[ClusteringSummarizer] Loaded sentence transformer")
            except ImportError:
                print("[ClusteringSummarizer] sentence-transformers not available")
                self.embedder = False
    
    def summarize(
        self,
        text: str,
        num_sentences: int = 3
    ) -> Tuple[str, List[str]]:
        """
        Summarize using sentence clustering
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            
        Returns:
            Tuple of (summary, key sentences)
        """
        sentences = self._split_sentences(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences), sentences
        
        self._load_embedder()
        
        if self.embedder and self.embedder is not False:
            return self._cluster_with_embeddings(sentences, num_sentences)
        else:
            return self._cluster_with_tfidf(sentences, num_sentences)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _cluster_with_embeddings(
        self,
        sentences: List[str],
        num_clusters: int
    ) -> Tuple[str, List[str]]:
        """Cluster using sentence embeddings"""
        embeddings = self.embedder.encode(sentences)
        
        # Simple k-means clustering
        centroids, labels = self._kmeans(embeddings, num_clusters)
        
        # Select sentence closest to each centroid
        selected_indices = []
        for cluster_id in range(num_clusters):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                continue
            
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Find closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroids[cluster_id], axis=1)
            best_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(best_idx)
        
        # Sort by original order
        selected_indices = sorted(selected_indices)
        key_sentences = [sentences[i] for i in selected_indices]
        
        return ' '.join(key_sentences), key_sentences
    
    def _cluster_with_tfidf(
        self,
        sentences: List[str],
        num_clusters: int
    ) -> Tuple[str, List[str]]:
        """Fallback clustering using TF-IDF"""
        # Build vocabulary
        vocab = {}
        for sent in sentences:
            for word in re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower()):
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        # Build TF-IDF matrix
        n_docs = len(sentences)
        n_terms = len(vocab)
        
        if n_terms == 0:
            return sentences[0], [sentences[0]]
        
        tfidf = np.zeros((n_docs, n_terms))
        
        # Term frequency
        for i, sent in enumerate(sentences):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
            for word in words:
                if word in vocab:
                    tfidf[i, vocab[word]] += 1
        
        # IDF
        doc_freq = (tfidf > 0).sum(axis=0)
        idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1
        tfidf = tfidf * idf
        
        # Normalize
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        tfidf = tfidf / norms
        
        # Cluster
        centroids, labels = self._kmeans(tfidf, num_clusters)
        
        # Select representatives
        selected_indices = []
        for cluster_id in range(num_clusters):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                continue
            
            cluster_indices = np.where(cluster_mask)[0]
            # Pick first sentence in cluster (usually most relevant)
            selected_indices.append(cluster_indices[0])
        
        selected_indices = sorted(selected_indices)
        key_sentences = [sentences[i] for i in selected_indices]
        
        return ' '.join(key_sentences), key_sentences
    
    def _kmeans(
        self,
        data: np.ndarray,
        k: int,
        max_iters: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering"""
        n = data.shape[0]
        k = min(k, n)
        
        # Initialize centroids randomly
        indices = np.random.choice(n, k, replace=False)
        centroids = data[indices].copy()
        
        labels = np.zeros(n, dtype=int)
        
        for _ in range(max_iters):
            # Assign labels
            for i in range(n):
                distances = np.linalg.norm(data[i] - centroids, axis=1)
                labels[i] = np.argmin(distances)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = data[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return centroids, labels


class TransformerSummarizer:
    """
    Transformer-based abstractive summarization
    
    Uses BART or T5 for generating summaries.
    """
    
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """
        Initialize transformer summarizer
        
        Args:
            model_name: HuggingFace model name
                - "facebook/bart-large-cnn" (better quality, larger)
                - "sshleifer/distilbart-cnn-12-6" (faster, smaller)
                - "t5-small" (lightweight)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Load the transformer model"""
        if self.model is not None:
            return True
        
        try:
            from transformers import pipeline
            print(f"[TransformerSummarizer] Loading {self.model_name}...")
            self.model = pipeline("summarization", model=self.model_name)
            print("[TransformerSummarizer] Model loaded")
            return True
        except ImportError:
            print("[TransformerSummarizer] transformers library not available")
            print("  Install with: pip install transformers torch")
            return False
        except Exception as e:
            print(f"[TransformerSummarizer] Error loading model: {e}")
            return False
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30
    ) -> Tuple[str, float]:
        """
        Generate abstractive summary
        
        Args:
            text: Input text
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Tuple of (summary, confidence)
        """
        if not self._load_model():
            return "", 0.0
        
        # Truncate very long texts
        if len(text) > 10000:
            text = text[:10000]
        
        try:
            result = self.model(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summary = result[0]['summary_text']
            return summary, 0.8
        except Exception as e:
            print(f"[TransformerSummarizer] Error: {e}")
            return "", 0.0


class MeetingPatternSummarizer:
    """
    Meeting-specific pattern extraction
    
    Extracts key meeting elements:
    - Decisions made
    - Action items
    - Key discussion points
    """
    
    # Patterns for key meeting elements
    DECISION_PATTERNS = [
        r'we\s+(?:have\s+)?decided\s+(?:to\s+)?(.+?)(?:[.!]|$)',
        r'decision\s*(?:is|:)?\s*(.+?)(?:[.!]|$)',
        r'agreed\s+(?:to|on|that)\s+(.+?)(?:[.!]|$)',
        r'final(?:ly)?\s+(?:we|decision)\s+(.+?)(?:[.!]|$)',
    ]
    
    ACTION_PATTERNS = [
        r'(\w+)\s+will\s+(.+?)(?:[.!]|$)',
        r'we\s+need\s+to\s+(.+?)(?:[.!]|$)',
        r'action\s*item[:\s]+(.+?)(?:[.!]|$)',
        r'next\s+step[s]?\s*(?:is|are|:)?\s*(.+?)(?:[.!]|$)',
    ]
    
    TOPIC_PATTERNS = [
        r'(?:discussed|talking)\s+about\s+(.+?)(?:[.!,]|$)',
        r'regarding\s+(.+?)(?:[.!,]|$)',
        r'topic\s+(?:is|:)?\s*(.+?)(?:[.!]|$)',
        r'agenda\s+(?:item|:)?\s*(.+?)(?:[.!]|$)',
    ]
    
    def summarize(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract meeting-specific summary
        
        Args:
            text: Meeting transcript
            
        Returns:
            Tuple of (formatted summary, key points)
        """
        decisions = self._extract_patterns(text, self.DECISION_PATTERNS)
        actions = self._extract_patterns(text, self.ACTION_PATTERNS)
        topics = self._extract_patterns(text, self.TOPIC_PATTERNS)
        
        # Build summary
        parts = []
        key_points = []
        
        if topics:
            parts.append("Topics discussed: " + ", ".join(topics[:3]))
            key_points.extend(topics[:3])
        
        if decisions:
            parts.append("Decisions: " + "; ".join(decisions[:3]))
            key_points.extend(decisions[:3])
        
        if actions:
            parts.append("Action items: " + "; ".join(actions[:3]))
            key_points.extend(actions[:3])
        
        summary = ". ".join(parts) if parts else ""
        return summary, key_points
    
    def _extract_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Extract matches for patterns"""
        results = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get the captured group
                if match.groups():
                    captured = match.group(len(match.groups()))
                    if captured and len(captured) > 5:
                        results.append(captured.strip())
        return list(set(results))[:5]  # Deduplicate and limit


class HybridSummarizer:
    """
    Hybrid Summarization Engine
    
    TECHNICAL NOVELTY:
    Combines multiple summarization strategies:
    1. Transformer (abstractive) - generates new text
    2. Clustering (extractive) - semantic grouping
    3. TextRank (extractive) - graph-based ranking
    4. Meeting patterns - domain-specific extraction
    
    Selects best strategy based on:
    - Content type (meeting vs general)
    - Text length
    - Available models
    
    Usage:
        summarizer = HybridSummarizer()
        result = summarizer.summarize(transcript_text)
        
        print(result.summary)
        print(f"Strategy: {result.strategy_used}")
        print(f"Key points: {result.key_points}")
    """
    
    def __init__(
        self,
        use_transformer: bool = True,
        transformer_model: str = "sshleifer/distilbart-cnn-12-6"
    ):
        """
        Initialize hybrid summarizer
        
        Args:
            use_transformer: Whether to use transformer summarization
            transformer_model: HuggingFace model name
        """
        self.textrank = TextRankSummarizer()
        self.clustering = ClusteringSummarizer()
        self.meeting_patterns = MeetingPatternSummarizer()
        
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerSummarizer(transformer_model)
        else:
            self.transformer = None
    
    def summarize(
        self,
        text: str,
        strategy: str = "auto",
        max_length: int = 150,
        num_sentences: int = 5,
        is_meeting: bool = True
    ) -> SummaryResult:
        """
        Generate summary using hybrid approach
        
        Args:
            text: Input text
            strategy: "auto", "transformer", "textrank", "clustering", "meeting", or "ensemble"
            max_length: Max length for transformer summary
            num_sentences: Number of sentences for extractive
            is_meeting: Whether text is a meeting transcript
            
        Returns:
            SummaryResult with summary and metadata
        """
        original_word_count = len(text.split())
        
        if strategy == "auto":
            strategy = self._select_strategy(text, is_meeting)
        
        if strategy == "transformer" and self.transformer:
            summary, confidence = self.transformer.summarize(text, max_length)
            key_points = self._extract_key_points(summary)
            
        elif strategy == "clustering":
            summary, key_points = self.clustering.summarize(text, num_sentences)
            confidence = 0.75
            
        elif strategy == "meeting" and is_meeting:
            summary, key_points = self.meeting_patterns.summarize(text)
            confidence = 0.7
            
        elif strategy == "ensemble":
            summary, key_points, confidence = self._ensemble_summarize(
                text, max_length, num_sentences, is_meeting
            )
            
        else:  # textrank (default)
            summary, key_points = self.textrank.summarize(text, num_sentences)
            confidence = 0.7
        
        # Calculate compression ratio
        summary_word_count = len(summary.split())
        compression = summary_word_count / max(1, original_word_count)
        
        return SummaryResult(
            summary=summary,
            strategy_used=strategy,
            confidence=confidence,
            key_points=key_points[:5],
            word_count=summary_word_count,
            compression_ratio=compression
        )
    
    def _select_strategy(self, text: str, is_meeting: bool) -> str:
        """Auto-select best strategy based on content"""
        word_count = len(text.split())
        
        # Short text: use extractive
        if word_count < 100:
            return "textrank"
        
        # Meeting transcript: try meeting patterns first
        if is_meeting:
            meeting_summary, points = self.meeting_patterns.summarize(text)
            if len(points) >= 2:
                return "meeting"
        
        # Medium text with transformer available: use transformer
        if word_count < 2000 and self.transformer:
            return "transformer"
        
        # Long text: use ensemble
        if word_count > 500:
            return "ensemble"
        
        return "clustering"
    
    def _ensemble_summarize(
        self,
        text: str,
        max_length: int,
        num_sentences: int,
        is_meeting: bool
    ) -> Tuple[str, List[str], float]:
        """
        Ensemble summarization combining multiple methods
        
        Reduces hallucination by cross-validating extractive and abstractive.
        """
        summaries = []
        all_key_points = []
        
        # TextRank
        tr_summary, tr_points = self.textrank.summarize(text, num_sentences)
        summaries.append(('textrank', tr_summary, 0.7))
        all_key_points.extend(tr_points)
        
        # Clustering
        cl_summary, cl_points = self.clustering.summarize(text, num_sentences)
        summaries.append(('clustering', cl_summary, 0.75))
        all_key_points.extend(cl_points)
        
        # Transformer (if available)
        if self.transformer:
            tf_summary, tf_conf = self.transformer.summarize(text, max_length)
            if tf_summary:
                summaries.append(('transformer', tf_summary, tf_conf))
        
        # Meeting patterns
        if is_meeting:
            mt_summary, mt_points = self.meeting_patterns.summarize(text)
            if mt_summary:
                summaries.append(('meeting', mt_summary, 0.7))
                all_key_points.extend(mt_points)
        
        # Select best or combine
        if len(summaries) == 1:
            return summaries[0][1], all_key_points, summaries[0][2]
        
        # Prefer transformer if available, validated by extractive overlap
        transformer_summary = None
        for name, summary, conf in summaries:
            if name == 'transformer':
                transformer_summary = summary
                break
        
        if transformer_summary:
            # Validate transformer output against extractive
            extractive_words = set()
            for name, summary, _ in summaries:
                if name != 'transformer':
                    extractive_words.update(summary.lower().split())
            
            transformer_words = set(transformer_summary.lower().split())
            overlap = len(transformer_words & extractive_words) / max(1, len(transformer_words))
            
            if overlap > 0.3:  # Good overlap, use transformer
                return transformer_summary, all_key_points[:5], 0.85
        
        # Otherwise use clustering (semantic grouping)
        for name, summary, conf in summaries:
            if name == 'clustering':
                return summary, all_key_points[:5], conf
        
        # Fallback to textrank
        return summaries[0][1], all_key_points[:5], summaries[0][2]
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from summary text"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10][:5]
    
    def summarize_long_document(
        self,
        text: str,
        chunk_size: int = 1000,
        final_summary_length: int = 200
    ) -> SummaryResult:
        """
        Summarize long documents using hierarchical approach
        
        1. Split into chunks
        2. Summarize each chunk
        3. Combine and summarize again
        """
        words = text.split()
        
        if len(words) <= chunk_size:
            return self.summarize(text)
        
        # Split into chunks
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            result = self.summarize(chunk, strategy="textrank", num_sentences=3)
            chunk_summaries.append(result.summary)
        
        # Combine and summarize again
        combined = ' '.join(chunk_summaries)
        final_result = self.summarize(
            combined,
            strategy="ensemble" if self.transformer else "textrank",
            max_length=final_summary_length
        )
        
        final_result.strategy_used = "hierarchical_" + final_result.strategy_used
        return final_result
