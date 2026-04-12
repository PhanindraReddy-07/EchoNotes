"""
Sentence Importance Classifier - Custom ML Model
=================================================
A trainable neural network that classifies sentences as important or not.

THIS IS A CUSTOM TRAINED MODEL - Not pre-built!

Features extracted:
- Sentence length
- Position in document
- TF-IDF score
- Contains numbers
- Contains proper nouns
- Sentence structure features

Model: Simple feedforward neural network trained from scratch using NumPy.
No PyTorch/TensorFlow required!

Usage:
    # Training
    classifier = SentenceClassifier()
    classifier.train(training_data, labels, epochs=100)
    classifier.save_model('model.npz')
    
    # Inference
    classifier.load_model('model.npz')
    importance = classifier.predict(sentence, document_context)
"""

import numpy as np
import json
import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import math


@dataclass
class TrainingExample:
    """A single training example"""
    sentence: str
    document: str  # Full document for context
    position: int  # Position in document
    total_sentences: int
    is_important: int  # 1 = important, 0 = not important


class NeuralNetwork:
    """
    Simple feedforward neural network implemented from scratch.
    
    Architecture:
    - Input layer: n features
    - Hidden layer 1: 32 neurons (ReLU)
    - Hidden layer 2: 16 neurons (ReLU)
    - Output layer: 1 neuron (Sigmoid)
    
    NO PYTORCH/TENSORFLOW - Pure NumPy!
    """
    
    def __init__(self, input_size: int, learning_rate: float = 0.01):
        """Initialize network with random weights"""
        self.lr = learning_rate
        
        # Xavier initialization for weights
        self.W1 = np.random.randn(input_size, 32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, 32))
        
        self.W2 = np.random.randn(32, 16) * np.sqrt(2.0 / 32)
        self.b2 = np.zeros((1, 16))
        
        self.W3 = np.random.randn(16, 1) * np.sqrt(2.0 / 16)
        self.b3 = np.zeros((1, 1))
        
        # For batch normalization
        self.running_mean1 = np.zeros((1, 32))
        self.running_var1 = np.ones((1, 32))
        self.running_mean2 = np.zeros((1, 16))
        self.running_var2 = np.ones((1, 16))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative for backprop"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X: np.ndarray, training: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through the network
        
        Args:
            X: Input features (batch_size, input_size)
            training: Whether in training mode
            
        Returns:
            Output predictions and cache for backprop
        """
        cache = {}
        
        # Layer 1
        cache['Z1'] = X @ self.W1 + self.b1
        cache['A1'] = self.relu(cache['Z1'])
        
        # Layer 2
        cache['Z2'] = cache['A1'] @ self.W2 + self.b2
        cache['A2'] = self.relu(cache['Z2'])
        
        # Output layer
        cache['Z3'] = cache['A2'] @ self.W3 + self.b3
        cache['A3'] = self.sigmoid(cache['Z3'])
        
        cache['X'] = X
        
        return cache['A3'], cache
    
    def backward(self, y: np.ndarray, cache: dict) -> dict:
        """
        Backward pass - compute gradients
        
        Args:
            y: True labels
            cache: Cached values from forward pass
            
        Returns:
            Dictionary of gradients
        """
        m = y.shape[0]
        grads = {}
        
        # Output layer gradient
        dZ3 = cache['A3'] - y.reshape(-1, 1)
        grads['dW3'] = (cache['A2'].T @ dZ3) / m
        grads['db3'] = np.sum(dZ3, axis=0, keepdims=True) / m
        
        # Layer 2 gradient
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * self.relu_derivative(cache['Z2'])
        grads['dW2'] = (cache['A1'].T @ dZ2) / m
        grads['db2'] = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Layer 1 gradient
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(cache['Z1'])
        grads['dW1'] = (cache['X'].T @ dZ1) / m
        grads['db1'] = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return grads
    
    def update_weights(self, grads: dict):
        """Update weights using gradients"""
        self.W3 -= self.lr * grads['dW3']
        self.b3 -= self.lr * grads['db3']
        self.W2 -= self.lr * grads['dW2']
        self.b2 -= self.lr * grads['db2']
        self.W1 -= self.lr * grads['dW1']
        self.b1 -= self.lr * grads['db1']
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def get_weights(self) -> dict:
        """Get all weights as dictionary"""
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }
    
    def set_weights(self, weights: dict):
        """Set weights from dictionary"""
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        self.W3 = weights['W3']
        self.b3 = weights['b3']


class FeatureExtractor:
    """
    Extract features from sentences for ML model.
    
    Features (12 total):
    1. Normalized sentence length
    2. Normalized position in document
    3. Is first sentence (binary)
    4. Is last sentence (binary)
    5. Contains number (binary)
    6. Contains proper noun (binary)
    7. Average TF-IDF score
    8. Max TF-IDF score
    9. Sentence complexity (words per clause)
    10. Question sentence (binary)
    11. Contains keywords (binary)
    12. Unique word ratio
    """
    
    # Important indicator keywords
    IMPORTANT_KEYWORDS = {
        'important', 'key', 'main', 'significant', 'essential', 'crucial',
        'primary', 'critical', 'fundamental', 'major', 'central', 'vital',
        'notably', 'specifically', 'particularly', 'especially',
        'first', 'finally', 'conclusion', 'summary', 'result', 'therefore',
        'consequently', 'thus', 'hence', 'overall', 'in summary'
    }
    
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'just', 'also', 'now', 'here', 'there', 'then', 'once', 'if', 'as'
    }
    
    def __init__(self):
        self.tfidf_cache = {}
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [w for w in words if w not in self.STOPWORDS and len(w) > 2]
    
    def compute_tfidf(self, sentences: List[str]) -> Dict[str, float]:
        """Compute TF-IDF for document"""
        doc_freq = Counter()
        term_freq = Counter()
        
        for sent in sentences:
            words = set(self.tokenize(sent))
            doc_freq.update(words)
            term_freq.update(self.tokenize(sent))
        
        n_docs = len(sentences)
        tfidf = {}
        
        for word, tf in term_freq.items():
            df = doc_freq[word]
            idf = math.log(n_docs / (df + 1)) + 1
            tfidf[word] = tf * idf
        
        # Normalize
        if tfidf:
            max_score = max(tfidf.values())
            tfidf = {k: v / max_score for k, v in tfidf.items()}
        
        return tfidf
    
    def extract_features(
        self,
        sentence: str,
        document: str,
        position: int,
        total_sentences: int,
        tfidf: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Extract 12 features from a sentence.
        
        Returns:
            numpy array of 12 features
        """
        words = sentence.split()
        tokens = self.tokenize(sentence)
        
        # Compute TF-IDF if not provided
        if tfidf is None:
            sentences = re.split(r'[.!?]+', document)
            sentences = [s.strip() for s in sentences if s.strip()]
            tfidf = self.compute_tfidf(sentences)
        
        features = []
        
        # 1. Normalized sentence length (0-1)
        max_len = 50
        features.append(min(len(words) / max_len, 1.0))
        
        # 2. Normalized position (0-1)
        features.append(position / max(total_sentences - 1, 1))
        
        # 3. Is first sentence
        features.append(1.0 if position == 0 else 0.0)
        
        # 4. Is last sentence
        features.append(1.0 if position == total_sentences - 1 else 0.0)
        
        # 5. Contains number
        has_number = 1.0 if re.search(r'\d+', sentence) else 0.0
        features.append(has_number)
        
        # 6. Contains proper noun (capitalized word not at start)
        proper_nouns = re.findall(r'(?<!^)(?<![.!?]\s)[A-Z][a-z]+', sentence)
        features.append(1.0 if proper_nouns else 0.0)
        
        # 7. Average TF-IDF score
        if tokens and tfidf:
            avg_tfidf = sum(tfidf.get(t, 0) for t in tokens) / len(tokens)
        else:
            avg_tfidf = 0.0
        features.append(avg_tfidf)
        
        # 8. Max TF-IDF score
        if tokens and tfidf:
            max_tfidf = max(tfidf.get(t, 0) for t in tokens)
        else:
            max_tfidf = 0.0
        features.append(max_tfidf)
        
        # 9. Sentence complexity (approx words per clause)
        clauses = len(re.findall(r'[,;:]', sentence)) + 1
        complexity = len(words) / clauses / 15  # Normalize
        features.append(min(complexity, 1.0))
        
        # 10. Is question
        features.append(1.0 if sentence.strip().endswith('?') else 0.0)
        
        # 11. Contains important keywords
        sent_lower = sentence.lower()
        has_keyword = any(kw in sent_lower for kw in self.IMPORTANT_KEYWORDS)
        features.append(1.0 if has_keyword else 0.0)
        
        # 12. Unique word ratio
        if words:
            unique_ratio = len(set(w.lower() for w in words)) / len(words)
        else:
            unique_ratio = 0.0
        features.append(unique_ratio)
        
        return np.array(features, dtype=np.float32)


class SentenceImportanceClassifier:
    """
    ML-based sentence importance classifier.
    
    This is a CUSTOM TRAINED MODEL - the neural network weights
    are learned from training data, not pre-trained!
    
    Usage:
        # Create and train
        classifier = SentenceImportanceClassifier()
        classifier.train(sentences, documents, labels)
        
        # Save trained model
        classifier.save_model('sentence_model.npz')
        
        # Load and use
        classifier.load_model('sentence_model.npz')
        score = classifier.predict(sentence, document)
    """
    
    MODEL_DIR = Path.home() / ".cache" / "echonotes" / "models"
    DEFAULT_MODEL = "sentence_importance_v1.npz"
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize classifier"""
        self.feature_extractor = FeatureExtractor()
        self.network = None
        self.learning_rate = learning_rate
        self.is_trained = False
        self.training_history = []
        
        # Ensure model directory exists
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    def _prepare_training_data(
        self,
        examples: List[TrainingExample]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert training examples to feature matrix"""
        X = []
        y = []
        
        for ex in examples:
            # Get document sentences for TF-IDF
            sentences = re.split(r'[.!?]+', ex.document)
            sentences = [s.strip() for s in sentences if s.strip()]
            tfidf = self.feature_extractor.compute_tfidf(sentences)
            
            # Extract features
            features = self.feature_extractor.extract_features(
                ex.sentence,
                ex.document,
                ex.position,
                ex.total_sentences,
                tfidf
            )
            
            X.append(features)
            y.append(ex.is_important)
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        examples: List[TrainingExample],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> dict:
        """
        Train the neural network on labeled data.
        
        Args:
            examples: List of TrainingExample objects
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_split: Fraction for validation
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        if verbose:
            print("=" * 50)
            print("TRAINING SENTENCE IMPORTANCE CLASSIFIER")
            print("=" * 50)
            print(f"Examples: {len(examples)}")
            print(f"Epochs: {epochs}")
            print(f"Batch size: {batch_size}")
            print()
        
        # Prepare data
        if verbose:
            print("Extracting features...")
        X, y = self._prepare_training_data(examples)
        
        # Shuffle and split
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        if verbose:
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Features per sample: {X.shape[1]}")
            print()
        
        # Initialize network
        input_size = X.shape[1]
        self.network = NeuralNetwork(input_size, self.learning_rate)
        
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        # Training loop
        if verbose:
            print("Training...")
        
        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                
                # Forward pass
                y_pred, cache = self.network.forward(X_batch, training=True)
                
                # Compute loss
                loss = self.network.compute_loss(y_pred, y_batch.reshape(-1, 1))
                epoch_losses.append(loss)
                
                # Backward pass
                grads = self.network.backward(y_batch, cache)
                
                # Update weights
                self.network.update_weights(grads)
            
            # Compute metrics
            train_loss = np.mean(epoch_losses)
            
            # Validation
            val_pred, _ = self.network.forward(X_val)
            val_loss = self.network.compute_loss(val_pred, y_val.reshape(-1, 1))
            val_accuracy = np.mean((val_pred.flatten() > 0.5) == y_val)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_accuracy:.4f}")
        
        self.is_trained = True
        self.training_history = history
        
        if verbose:
            print()
            print("Training complete!")
            print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        
        return history
    
    def predict(
        self,
        sentence: str,
        document: str,
        position: int = 0,
        total_sentences: int = 1
    ) -> float:
        """
        Predict importance score for a sentence.
        
        Args:
            sentence: The sentence to classify
            document: Full document context
            position: Position in document
            total_sentences: Total sentences in document
            
        Returns:
            Importance score (0-1)
        """
        if not self.is_trained and self.network is None:
            # Try to load default model
            default_path = self.MODEL_DIR / self.DEFAULT_MODEL
            if default_path.exists():
                self.load_model(str(default_path))
            else:
                raise ValueError("Model not trained. Call train() first or load a model.")
        
        # Extract features
        sentences = re.split(r'[.!?]+', document)
        sentences = [s.strip() for s in sentences if s.strip()]
        tfidf = self.feature_extractor.compute_tfidf(sentences)
        
        features = self.feature_extractor.extract_features(
            sentence, document, position, total_sentences, tfidf
        )
        
        # Predict
        X = features.reshape(1, -1)
        prediction, _ = self.network.forward(X)
        
        return float(prediction[0, 0])
    
    def predict_batch(
        self,
        sentences: List[str],
        document: str
    ) -> List[Tuple[str, float]]:
        """
        Predict importance for all sentences in a document.
        
        Args:
            sentences: List of sentences
            document: Full document
            
        Returns:
            List of (sentence, score) tuples sorted by importance
        """
        # Compute TF-IDF once
        tfidf = self.feature_extractor.compute_tfidf(sentences)
        
        results = []
        for i, sent in enumerate(sentences):
            features = self.feature_extractor.extract_features(
                sent, document, i, len(sentences), tfidf
            )
            X = features.reshape(1, -1)
            score, _ = self.network.forward(X)
            results.append((sent, float(score[0, 0])))
        
        # Sort by importance
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def save_model(self, filepath: str = None):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if filepath is None:
            filepath = str(self.MODEL_DIR / self.DEFAULT_MODEL)
        
        weights = self.network.get_weights()
        np.savez(
            filepath,
            **weights,
            is_trained=np.array([1]),
            learning_rate=np.array([self.learning_rate])
        )
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load trained model from file"""
        if filepath is None:
            filepath = str(self.MODEL_DIR / self.DEFAULT_MODEL)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        data = np.load(filepath)
        
        # Determine input size from W1
        input_size = data['W1'].shape[0]
        
        # Initialize network
        self.learning_rate = float(data['learning_rate'][0])
        self.network = NeuralNetwork(input_size, self.learning_rate)
        
        # Load weights
        weights = {
            'W1': data['W1'], 'b1': data['b1'],
            'W2': data['W2'], 'b2': data['b2'],
            'W3': data['W3'], 'b3': data['b3']
        }
        self.network.set_weights(weights)
        
        self.is_trained = True
        print(f"Model loaded from: {filepath}")


def create_training_data_from_text(
    text: str,
    important_sentences: List[str]
) -> List[TrainingExample]:
    """
    Helper to create training data from labeled text.
    
    Args:
        text: Full document text
        important_sentences: List of sentences marked as important
        
    Returns:
        List of TrainingExample objects
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Create lowercase set for matching
    important_lower = {s.lower().strip() for s in important_sentences}
    
    examples = []
    for i, sent in enumerate(sentences):
        is_important = 1 if sent.lower().strip() in important_lower else 0
        
        examples.append(TrainingExample(
            sentence=sent,
            document=text,
            position=i,
            total_sentences=len(sentences),
            is_important=is_important
        ))
    
    return examples


def generate_synthetic_training_data(n_samples: int = 500) -> List[TrainingExample]:
    """
    Generate synthetic training data for initial model training.
    
    Creates examples with known patterns:
    - First sentences are often important
    - Sentences with numbers are often important
    - Sentences with keywords like "important", "key" are important
    - Very short sentences are usually not important
    """
    import random
    
    # Sample documents
    documents = [
        """Machine learning is a subset of artificial intelligence. It allows computers to learn from data.
        The first machine learning program was written in 1952. Today, ML is used in many applications.
        Neural networks are a key technology in modern ML. They can process complex patterns.
        The global ML market is worth $15 billion. It is expected to grow significantly.
        In conclusion, machine learning is transforming industries worldwide.""",
        
        """Instagram is a photo sharing social network. It was founded in 2010 by Kevin Systrom.
        The app allows users to share photos and videos. Users can apply filters to their content.
        Facebook acquired Instagram for $1 billion in 2012. This was a significant acquisition.
        Today, Instagram has over 2 billion active users. It is one of the most popular apps.
        The platform is important for digital marketing. Many businesses use it for promotion.""",
        
        """Climate change is affecting our planet. Global temperatures have risen by 1.1 degrees.
        The Paris Agreement aims to limit warming. It was signed by 196 countries in 2015.
        Renewable energy is crucial for reducing emissions. Solar and wind power are growing fast.
        Scientists warn that action is needed now. The next decade is critical for climate action.
        In summary, climate change requires immediate global cooperation.""",
        
        """Python is a popular programming language. It was created by Guido van Rossum in 1991.
        The language is known for its simple syntax. This makes it easy to learn.
        Python is used in web development and data science. It has many libraries available.
        Over 8 million developers use Python worldwide. It is one of the top 3 languages.
        Overall, Python continues to grow in popularity and importance.""",
    ]
    
    examples = []
    
    for doc in documents:
        sentences = re.split(r'(?<=[.!?])\s+', doc)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i, sent in enumerate(sentences):
            # Determine importance based on rules
            is_important = 0
            
            # First sentence often important
            if i == 0:
                is_important = 1
            # Last sentence (conclusion) often important
            elif i == len(sentences) - 1:
                is_important = 1
            # Contains numbers (statistics)
            elif re.search(r'\d+', sent):
                is_important = 1 if random.random() > 0.3 else 0
            # Contains important keywords
            elif any(kw in sent.lower() for kw in ['important', 'key', 'significant', 'crucial', 'major', 'conclusion', 'summary']):
                is_important = 1
            # Very short sentences usually not important
            elif len(sent.split()) < 5:
                is_important = 0
            # Random for others
            else:
                is_important = 1 if random.random() > 0.6 else 0
            
            examples.append(TrainingExample(
                sentence=sent,
                document=doc,
                position=i,
                total_sentences=len(sentences),
                is_important=is_important
            ))
    
    # Duplicate and add noise for more data
    while len(examples) < n_samples:
        ex = random.choice(examples)
        # Randomly flip some labels for noise
        label = ex.is_important
        if random.random() < 0.1:
            label = 1 - label
        
        examples.append(TrainingExample(
            sentence=ex.sentence,
            document=ex.document,
            position=ex.position,
            total_sentences=ex.total_sentences,
            is_important=label
        ))
    
    random.shuffle(examples)
    return examples[:n_samples]


# ============== Demo and Testing ==============

def demo_training():
    """Demonstrate model training"""
    print("=" * 60)
    print("SENTENCE IMPORTANCE CLASSIFIER - TRAINING DEMO")
    print("=" * 60)
    print()
    
    # Generate training data
    print("Generating synthetic training data...")
    training_data = generate_synthetic_training_data(500)
    print(f"Generated {len(training_data)} training examples")
    print()
    
    # Count class balance
    n_important = sum(1 for ex in training_data if ex.is_important == 1)
    print(f"Class balance: {n_important} important, {len(training_data) - n_important} not important")
    print()
    
    # Create and train classifier
    classifier = SentenceImportanceClassifier(learning_rate=0.01)
    
    history = classifier.train(
        training_data,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=True
    )
    
    # Save model
    classifier.save_model()
    
    return classifier


def demo_prediction(classifier: SentenceImportanceClassifier = None):
    """Demonstrate model prediction"""
    print()
    print("=" * 60)
    print("PREDICTION DEMO")
    print("=" * 60)
    print()
    
    if classifier is None:
        classifier = SentenceImportanceClassifier()
        classifier.load_model()
    
    # Test document
    test_doc = """
    Artificial Intelligence is transforming the modern world. 
    It enables machines to perform tasks that typically require human intelligence.
    The AI market is projected to reach $190 billion by 2025.
    Many companies are investing heavily in AI research.
    This is just a simple statement.
    In conclusion, AI represents one of the most important technological advances of our time.
    """
    
    sentences = re.split(r'(?<=[.!?])\s+', test_doc)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print("Test document sentences ranked by importance:")
    print("-" * 50)
    
    results = classifier.predict_batch(sentences, test_doc)
    
    for i, (sent, score) in enumerate(results, 1):
        importance = "HIGH" if score > 0.6 else "MEDIUM" if score > 0.4 else "LOW"
        print(f"{i}. [{importance}] (score: {score:.3f})")
        print(f"   {sent[:70]}...")
        print()


if __name__ == "__main__":
    # Train model
    classifier = demo_training()
    
    # Test predictions
    demo_prediction(classifier)
