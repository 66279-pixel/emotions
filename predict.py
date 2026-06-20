"""
predict.py — Loads the trained model + vectorizer and predicts the emotion of new text.

Usage as a script:
    python src/predict.py "I can't believe we won the championship!"

Usage as a module:
    from predict import load_model, predict_emotion
    model, vectorizer = load_model()
    label, probs = predict_emotion("I'm so excited!", model, vectorizer)
"""

import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import clean_text  # noqa: E402

EMOJI_MAP = {
    "joy": "😄", "sadness": "😢", "anger": "😠",
    "fear": "😨", "surprise": "😲", "neutral": "😐",
}

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model(model_dir: str = DEFAULT_MODEL_DIR):
    """Loads the trained model and TF-IDF vectorizer from disk."""
    model_path = os.path.join(model_dir, "best_emotion_model.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Model files not found in {model_dir}/. Run `python src/train.py` first."
        )
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_emotion(text: str, model, vectorizer):
    """Returns (predicted_label, {label: probability, ...}) for a piece of text."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        prob_dict = {label: float(p) for label, p in zip(model.classes_, probs)}
    else:
        prob_dict = {label: (1.0 if label == pred else 0.0) for label in model.classes_}
    return pred, prob_dict


def main():
    model, vectorizer = load_model()
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, probs = predict_emotion(text, model, vectorizer)
        emoji = EMOJI_MAP.get(label, "")
        print(f"Text: {text}")
        print(f"Predicted: {emoji} {label}  (confidence: {probs[label]:.1%})")
        print("\nAll probabilities:")
        for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k:10s} {v:.1%}")
    else:
        print("Usage: python src/predict.py \"your sentence here\"")


if __name__ == "__main__":
    main()
