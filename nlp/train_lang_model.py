"""
EchoNotes Language ID Classifier — Training Script
====================================================
Run this once to train and save the language classifier.

Usage:
    python nlp/train_lang_model.py

What it does:
    1. Loads training data from nlp/lang_data.py
    2. Trains TF-IDF + Logistic Regression classifier
    3. Evaluates with cross-validation
    4. Saves model to models/lang_id.pkl
    5. Runs a quick self-test to verify everything works

After training, the model is automatically used by the pipeline.
No changes to other files needed — lang_classifier.py loads it
on startup.
"""

import sys
from pathlib import Path

# Make sure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def train():
    print("=" * 60)
    print("  EchoNotes Language ID Classifier — Training")
    print("=" * 60)
    print()

    # ── Step 1: Load training data ───────────────────────────────────
    print("📂 Step 1: Loading training data...")
    try:
        from lang_data import get_training_data, get_label_info, ENGLISH, HINDI, TELUGU, HINDI_ENGLISH, TELUGU_ENGLISH
    except ImportError:
        from nlp.lang_data import get_training_data, get_label_info

    texts, labels = get_training_data()
    label_info = get_label_info()

    print(f"   Total samples: {len(texts)}")
    from collections import Counter
    dist = Counter(labels)
    for lbl in ['en', 'hi', 'te', 'hi_en', 'te_en']:
        cnt = dist.get(lbl, 0)
        name = label_info.get(lbl, lbl)
        bar = '█' * cnt
        print(f"   {lbl:8s} {name:30s} {cnt:3d}  {bar}")
    print()

    # ── Step 2: Train ────────────────────────────────────────────────
    print("🧠 Step 2: Training classifier...")
    try:
        from lang_classifier import LanguageClassifier
    except ImportError:
        from nlp.lang_classifier import LanguageClassifier

    clf = LanguageClassifier()
    results = clf.train(texts, labels, verbose=True)
    print()

    # ── Step 3: Save ─────────────────────────────────────────────────
    print("💾 Step 3: Saving model...")
    clf.save()
    print()

    # ── Step 4: Self-test ────────────────────────────────────────────
    print("✅ Step 4: Self-test (10 examples)...")
    print()

    test_cases = [
        # (text, expected_label)
        ("machine learning is a subset of artificial intelligence", "en"),
        ("today we will learn about data structures and algorithms", "en"),
        ("मशीन लर्निंग एक ऐसी तकनीक है जो डेटा से सीखती है", "hi"),
        ("ऑपरेटिंग सिस्टम हार्डवेयर और सॉफ्टवेयर को नियंत्रित करता है", "hi"),
        ("మెషిన్ లెర్నింగ్ అనేది కంప్యూటర్ సిస్టమ్‌లు డేటా నుండి నేర్చుకునే సామర్థ్యం", "te"),
        ("ఈరోజు మనం ఆపరేటింగ్ సిస్టమ్‌ల గురించి నేర్చుకుంటాం", "te"),
        ("aaj hum machine learning ke baare mein padenge aur uske applications dekhenge", "hi_en"),
        ("yeh algorithm bahut efficient hai kyunki time complexity order n log n hai", "hi_en"),
        ("nenu ippudu machine learning gurinchi chepputhanu mee kosam", "te_en"),
        ("ee algorithm time complexity order n log n ga untundi chala efficient", "te_en"),
    ]

    passed = 0
    for text, expected in test_cases:
        pred = clf.predict(text)
        ok = pred.label == expected
        passed += ok
        icon = "✅" if ok else "❌"
        print(f"  {icon}  Expected: {expected:8s}  Got: {pred.label:8s}  Conf: {pred.confidence:.2f}")
        if not ok:
            print(f"      Text: {text[:60]}...")
    print()
    print(f"  Self-test: {passed}/{len(test_cases)} passed")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Cross-val accuracy : {results['accuracy']:.1%}")
    print(f"  Self-test accuracy : {passed/len(test_cases):.1%}")
    print(f"  Model saved to     : models/lang_id.pkl")
    print()
    print("  The classifier is now used automatically by:")
    print("    - smart_analyzer.py   (script detection)")
    print("    - content_enhancer.py (sentence splitting)")
    print("    - smart_generator.py  (PDF font selection)")
    print("    - code_mix_handler.py (language segments)")
    print("=" * 60)

    # ── Step 5: Show how to integrate ────────────────────────────────
    print()
    print("📋 HOW TO USE IN YOUR CODE:")
    print()
    print("  from nlp.lang_classifier import detect_language, get_classifier")
    print()
    print("  # Simple detection")
    print("  lang = detect_language(transcript_text)")
    print("  # Returns: 'en', 'hi', 'te', 'hi_en', 'te_en'")
    print()
    print("  # Full prediction with confidence")
    print("  clf = get_classifier()")
    print("  result = clf.predict(transcript_text)")
    print("  print(result.label)          # 'te_en'")
    print("  print(result.is_telugu)      # True")
    print("  print(result.primary_script) # 'telugu'")
    print("  print(result.confidence)     # 0.94")
    print()
    print("  # In smart_generator.py _generate_pdf:")
    print("  # REPLACE: _script = _detect_script(sample)")
    print("  # WITH:    _script = get_classifier().predict(sample).primary_script")


if __name__ == "__main__":
    train()
