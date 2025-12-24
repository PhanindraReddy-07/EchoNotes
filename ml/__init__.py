"""
EchoNotes ML Module
====================

Custom trained machine learning models.

Models:
- SentenceImportanceClassifier: Neural network to classify sentence importance

Datasets:
- CNN/DailyMail: News articles with human-written highlights
- DUC 2002: Document Understanding Conference extractive summaries

These models are TRAINED FROM SCRATCH on real datasets - not pre-trained!

Usage:
    from ml.sentence_classifier import SentenceImportanceClassifier
    from ml.dataset_loader import create_combined_dataset
    
    # Load real training data
    training_data = create_combined_dataset(n_cnn=50, n_duc=30)
    
    # Train a new model
    classifier = SentenceImportanceClassifier()
    classifier.train(training_data, epochs=100)
    classifier.save_model('model.npz')
    
    # Use trained model
    classifier.load_model('model.npz')
    score = classifier.predict(sentence, document)
"""

from .sentence_classifier import (
    SentenceImportanceClassifier,
    TrainingExample,
    FeatureExtractor,
    NeuralNetwork,
    generate_synthetic_training_data,
    create_training_data_from_text
)

from .dataset_loader import (
    DatasetLoader,
    DatasetInfo,
    create_combined_dataset
)

__all__ = [
    'SentenceImportanceClassifier',
    'TrainingExample',
    'FeatureExtractor',
    'NeuralNetwork',
    'generate_synthetic_training_data',
    'create_training_data_from_text',
    'DatasetLoader',
    'DatasetInfo',
    'create_combined_dataset'
]
