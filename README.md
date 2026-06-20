# emotions
# Text Emotion Prediction — Project Structure

A complete 6-class text emotion prediction project (`joy`, `sadness`, `anger`, `fear`, `surprise`, `neutral`) with model comparison/selection and a real-time Gradio web interface.

## Directory structure

```
emotion_project/
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # text cleaning
│   ├── dataset.py         # builds the labeled dataset + a separate "novel" eval set
│   ├── train.py           # trains/compares models, selects the best, saves it
│   ├── predict.py         # loads the saved model and predicts on new text
│   └── app.py              # Gradio real-time web interface
├── data/
│   └── emotion_data.csv   # generated training dataset
├── models/
│   ├── best_emotion_model.pkl     # already trained — ready to use immediately
│   └── tfidf_vectorizer.pkl
└── outputs/                # confusion matrix plot, etc.
```

## Quick start (already-trained model included)

```bash
pip install -r requirements.txt

# Predict from the command line
python src/predict.py "I can't believe we won the championship!"

# Launch the real-time Gradio web app
python src/app.py
```

## Retrain from scratch

```bash
python src/train.py
```

This rebuilds the dataset, compares Logistic Regression / Linear SVM / Multinomial Naive Bayes / Random Forest, picks the best by held-out test accuracy, evaluates on a completely separate set of 48 never-seen sentences (`novel_eval` in `src/dataset.py`), and overwrites `models/*.pkl`.

Both the held-out test accuracy and the novel-sentence accuracy are asserted to be **≥80%** — this checks genuine generalization, not just memorization of the training data's phrasing patterns.

## Run in Google Colab

Use the companion notebook `emotion_project_colab.ipynb` — it scaffolds this exact directory structure inside the Colab runtime (writing each `src/*.py` file to disk), then runs the full pipeline and launches the Gradio app with a public link. Just open it in Colab and `Runtime → Run all`.

You can also upload this whole `emotion_project/` folder directly to a Colab session (e.g. via the file browser or Google Drive) and run the modules from there instead.

## Extending

- **Bigger dataset:** replace the dataset-building logic in `src/dataset.py`'s `build_dataset()` with a `pd.read_csv(...)` on a larger real-world dataset (e.g. the Kaggle "Emotions dataset for NLP" or HuggingFace `emotion` dataset). Everything downstream works unchanged as long as the dataframe has `text` and `label` columns.
- **More classes:** extend the base sentence lists (`joy_base`, `sadness_base`, etc.) in `src/dataset.py`.
- **Higher accuracy ceiling:** fine-tune a transformer (e.g. DistilBERT) instead of TF-IDF + linear models — more compute, but better handling of longer/subtler text.
- **Permanent deployment:** copy `src/`, `models/`, and `requirements.txt` to a Hugging Face Space — `src/app.py` is already a working Gradio entry point.
