"""
train.py — Trains and compares multiple classifiers on TF-IDF features,
selects the best model, evaluates it (held-out test set + novel sentences),
and saves the model + vectorizer to disk.

Run directly:  python src/train.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_dataset, novel_eval          # noqa: E402
from preprocessing import clean_text                    # noqa: E402

RANDOM_STATE = 42
ACCURACY_TARGET = 0.80


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10, random_state=RANDOM_STATE),
        "Linear SVM": LinearSVC(C=1, random_state=RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def train_and_select_best(df: pd.DataFrame, verbose: bool = True):
    """Trains all candidate models, compares them, and returns the best one.

    Returns a dict with: best_model_name, best_model, best_acc, results_df,
    tfidf, X_test_tfidf, y_test.
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000, sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    results = []
    trained_models = {}
    for name, clf in get_models().items():
        clf.fit(X_train_tfidf, y_train)
        preds = clf.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring="accuracy")
        results.append({
            "Model": name,
            "Test Accuracy": round(acc, 4),
            "Macro F1": round(f1, 4),
            "CV Mean Accuracy": round(cv_scores.mean(), 4),
            "CV Std": round(cv_scores.std(), 4),
        })
        trained_models[name] = clf
        if verbose:
            print(f"{name:25s} test_acc={acc:.4f}  macro_f1={f1:.4f}  cv_mean={cv_scores.mean():.4f}")

    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_acc = results_df.iloc[0]["Test Accuracy"]

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_acc": best_acc,
        "results_df": results_df,
        "tfidf": tfidf,
        "X_test_tfidf": X_test_tfidf,
        "y_test": y_test,
    }


def evaluate_on_novel(model, tfidf):
    """Evaluates a trained model on the held-out novel_eval sentences (never used in training)."""
    novel_X = [clean_text(t) for t, _ in novel_eval]
    novel_y = [label for _, label in novel_eval]
    novel_X_tfidf = tfidf.transform(novel_X)
    novel_pred = model.predict(novel_X_tfidf)
    novel_acc = accuracy_score(novel_y, novel_pred)
    return novel_acc, novel_y, novel_pred


def ensure_proba_model(best_model, X_train_tfidf=None, y_train=None, X_test_tfidf=None, y_test=None):
    """Wraps a model lacking predict_proba (e.g. LinearSVC) with CalibratedClassifierCV."""
    if hasattr(best_model, "predict_proba"):
        return best_model
    calibrated = CalibratedClassifierCV(best_model, cv=5)
    calibrated.fit(X_train_tfidf, y_train)
    return calibrated


def main():
    print("Building dataset...")
    df = build_dataset()
    print("Dataset shape:", df.shape)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(os.path.join(data_dir, "emotion_data.csv"), index=False)
    print(f"Saved dataset to {data_dir}/emotion_data.csv")

    print("\nTraining and comparing models...")
    result = train_and_select_best(df)

    print(f"\nBest model: {result['best_model_name']}  (held-out test acc: {result['best_acc']:.2%})")
    assert result["best_acc"] >= ACCURACY_TARGET, "Held-out accuracy below 80% target."

    y_pred = result["best_model"].predict(result["X_test_tfidf"])
    print("\nClassification report (held-out test set):")
    print(classification_report(result["y_test"], y_pred))
    cm = confusion_matrix(result["y_test"], y_pred, labels=sorted(df["label"].unique()))
    print("Confusion matrix:")
    print(cm)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=sorted(df["label"].unique()), yticklabels=sorted(df["label"].unique()))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix — {result['best_model_name']} (held-out test set)")
        plt.savefig(os.path.join(outputs_dir, "confusion_matrix.png"), bbox_inches="tight", dpi=120)
        plt.close()
        print(f"Saved confusion matrix plot to {outputs_dir}/confusion_matrix.png")
    except ImportError:
        pass

    novel_acc, novel_y, novel_pred = evaluate_on_novel(result["best_model"], result["tfidf"])
    print(f"\nNovel-sentence accuracy (never seen in training): {novel_acc:.2%}")
    assert novel_acc >= ACCURACY_TARGET, "Novel-sentence accuracy below 80% target."
    print("classification report (novel sentences):")
    print(classification_report(novel_y, novel_pred))

    # Re-fit train split to get train tfidf/y_train for calibration if needed
    df2 = df.copy()
    df2["clean_text"] = df2["text"].apply(clean_text)
    from sklearn.model_selection import train_test_split as tts
    X_train, X_test, y_train, y_test = tts(
        df2["clean_text"], df2["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df2["label"]
    )
    X_train_tfidf = result["tfidf"].transform(X_train)
    proba_model = ensure_proba_model(
        result["best_model"], X_train_tfidf, y_train, result["X_test_tfidf"], result["y_test"]
    )

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(proba_model, os.path.join(models_dir, "best_emotion_model.pkl"))
    joblib.dump(result["tfidf"], os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    print(f"\nSaved model + vectorizer to {models_dir}/")

    return result, proba_model


if __name__ == "__main__":
    main()
