"""
app.py — Real-time Gradio web interface for the emotion prediction model.

Run directly:
    python src/app.py

In Google Colab, this launches a public shareable link (share=True).
Locally, it launches at http://127.0.0.1:7860 by default.
"""

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))
from predict import EMOJI_MAP, load_model, predict_emotion  # noqa: E402


def build_interface(model, vectorizer, held_out_acc=None, novel_acc=None):
    def gradio_predict(text):
        if not text or not text.strip():
            return "Please enter a sentence.", {}
        label, probs = predict_emotion(text, model, vectorizer)
        emoji = EMOJI_MAP.get(label, "")
        result_label = f"{emoji} {label.upper()}"
        sorted_probs = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
        return result_label, sorted_probs

    acc_line = ""
    if held_out_acc is not None:
        acc_line += f"Held-out test accuracy: **{held_out_acc:.1%}** &nbsp;|&nbsp; "
    if novel_acc is not None:
        acc_line += f"Novel-sentence accuracy: **{novel_acc:.1%}** &nbsp;|&nbsp; "

    demo = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Textbox(
            lines=3,
            placeholder="Type a sentence... e.g. 'I can't believe we won the championship!'",
            label="Enter your text",
        ),
        outputs=[
            gr.Label(label="Predicted Emotion"),
            gr.Label(label="Confidence per Emotion", num_top_classes=6),
        ],
        title="🎭 Real-Time Text Emotion Prediction",
        description=(
            f"Model: **{type(model).__name__}** trained on TF-IDF features &nbsp;|&nbsp; "
            f"{acc_line}Classes: joy, sadness, anger, fear, surprise, neutral.\n\n"
            "Type any sentence below and the model predicts its emotion in real time."
        ),
        examples=[
            ["I am beyond excited for the trip tomorrow!"],
            ["I feel so lonely and empty since they left."],
            ["This is unacceptable, I demand a refund right now!"],
            ["I'm scared we might lose everything in the storm."],
            ["Wait, seriously? I had absolutely no idea!"],
            ["The invoice was sent to the finance department."],
        ],
        live=False,
    )
    return demo


def main():
    model, vectorizer = load_model()
    demo = build_interface(model, vectorizer)
    # theme is passed to launch() for compatibility across Gradio 4/5/6
    demo.launch(share=True, debug=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
