#!/usr/bin/env python3
"""
Train Sentence Importance Classifier
=====================================

This script trains the custom ML model on REAL datasets.

Datasets used:
- CNN/DailyMail: News articles with human-written highlights
- DUC 2002: Document Understanding Conference extractive summaries

The key insight: Sentences that humans selected for summaries = "important"

Usage:
    python train_model.py                    # Train with real sample data
    python train_model.py --synthetic        # Use synthetic data (for testing)
    python train_model.py --data data.json   # Train with custom data
    python train_model.py --epochs 200       # Custom epochs

The trained model is saved to ~/.cache/echonotes/models/
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.sentence_classifier import (
    SentenceImportanceClassifier,
    TrainingExample,
    generate_synthetic_training_data,
    create_training_data_from_text
)
from ml.dataset_loader import (
    DatasetLoader,
    create_combined_dataset
)


def load_custom_data(filepath: str):
    """
    Load custom training data from JSON file.
    
    Expected format:
    {
        "documents": [
            {
                "text": "Full document text...",
                "important_sentences": [
                    "First important sentence.",
                    "Second important sentence."
                ]
            }
        ]
    }
    
    OR the format saved by download_dataset.py (list of examples)
    """
    # Expand ~ for Windows compatibility
    filepath = os.path.expanduser(filepath)
    
    loader = DatasetLoader()
    
    # Check if it's our downloaded format or custom format
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # If it's a list, it's from download_dataset.py
    if isinstance(data, list):
        examples = []
        for item in data:
            examples.append(TrainingExample(
                sentence=item['sentence'],
                document=item['document'],
                position=item['position'],
                total_sentences=item['total_sentences'],
                is_important=item['is_important']
            ))
        print(f"Loaded {len(examples)} examples from {filepath}")
        return examples
    else:
        # Custom format with documents
        return loader.load_custom_dataset(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Train Sentence Importance Classifier on Real Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
    By default, trains on CNN/DailyMail and DUC 2002 sample data.
    These are real extractive summarization datasets where humans
    have already identified important sentences.

Examples:
    python train_model.py                      # Train with real data
    python train_model.py --synthetic          # Use synthetic data
    python train_model.py --epochs 200         # More epochs
    python train_model.py --data custom.json   # Use custom data
    python train_model.py --cnn 100 --duc 50   # More samples
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to custom training data JSON file'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data instead of real datasets'
    )
    parser.add_argument(
        '--cnn',
        type=int,
        default=50,
        help='Number of CNN/DailyMail articles (default: 50)'
    )
    parser.add_argument(
        '--duc',
        type=int,
        default=30,
        help='Number of DUC documents (default: 30)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output model filename (default: sentence_importance_v1.npz)'
    )
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balance classes by undersampling majority class'
    )
    parser.add_argument(
        '--max-examples',
        type=int,
        default=None,
        help='Maximum total examples to use (for memory/speed)'
    )
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     SENTENCE IMPORTANCE CLASSIFIER                        ║
    ║     Training on Real Datasets                             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Load training data
    if args.data:
        print(f"Loading custom training data from: {args.data}")
        training_data = load_custom_data(args.data)
        data_source = "Custom dataset"
    elif args.synthetic:
        print("Using synthetic training data...")
        training_data = generate_synthetic_training_data(500)
        data_source = "Synthetic data"
    else:
        print("Loading real datasets...")
        print(f"  - CNN/DailyMail: {args.cnn} articles")
        print(f"  - DUC 2002: {args.duc} documents")
        print()
        training_data = create_combined_dataset(n_cnn=args.cnn, n_duc=args.duc)
        data_source = "CNN/DailyMail + DUC 2002"
    
    print()
    print(f"Data source: {data_source}")
    print(f"Total training examples: {len(training_data)}")
    
    # Count class balance
    n_important = sum(1 for ex in training_data if ex.is_important == 1)
    n_not_important = len(training_data) - n_important
    print(f"Class balance: {n_important} important ({n_important/len(training_data)*100:.1f}%), "
          f"{n_not_important} not important ({n_not_important/len(training_data)*100:.1f}%)")
    
    # Balance classes if requested
    if args.balance and n_important > 0:
        print("\n⚖️  Balancing classes...")
        import random
        
        # Separate by class
        important = [ex for ex in training_data if ex.is_important == 1]
        not_important = [ex for ex in training_data if ex.is_important == 0]
        
        # Undersample majority class to match minority (with some extra)
        target_size = min(len(not_important), len(important) * 3)  # 1:3 ratio
        random.shuffle(not_important)
        not_important = not_important[:target_size]
        
        training_data = important + not_important
        random.shuffle(training_data)
        
        n_important = len(important)
        n_not_important = len(not_important)
        print(f"After balancing: {n_important} important ({n_important/len(training_data)*100:.1f}%), "
              f"{n_not_important} not important ({n_not_important/len(training_data)*100:.1f}%)")
    
    # Limit examples if requested
    if args.max_examples and len(training_data) > args.max_examples:
        import random
        random.shuffle(training_data)
        training_data = training_data[:args.max_examples]
        print(f"Limited to {args.max_examples} examples")
    
    print()
    
    # Create classifier
    classifier = SentenceImportanceClassifier(learning_rate=args.lr)
    
    # Train
    print("Starting training...")
    print("-" * 50)
    
    history = classifier.train(
        training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=True
    )
    
    # Save model
    if args.output:
        output_path = str(classifier.MODEL_DIR / f"{args.output}.npz")
    else:
        output_path = None
    
    classifier.save_model(output_path)
    
    # Print summary
    print()
    print("=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Data source: {data_source}")
    print(f"Training examples: {len(training_data)}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print()
    print("Model saved successfully!")
    print()
    
    # Test prediction
    print("Testing model with sample prediction...")
    print("-" * 50)
    
    test_sentences = [
        "Apple reported revenue of $89.5 billion this quarter.",
        "The weather was nice yesterday.",
        "Scientists discovered a new species in the deep ocean.",
        "In conclusion, this represents a major breakthrough.",
    ]
    
    test_doc = " ".join(test_sentences)
    
    for i, sent in enumerate(test_sentences):
        score = classifier.predict(sent, test_doc, i, len(test_sentences))
        importance = "HIGH" if score > 0.6 else "MEDIUM" if score > 0.4 else "LOW"
        print(f"[{importance}] ({score:.3f}) {sent}")
    
    print()
    print("Training complete!")


if __name__ == "__main__":
    main()
