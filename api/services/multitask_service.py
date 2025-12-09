# api/services/multitask_service.py
import torch
from pathlib import Path
from model.multitask_model import (
    load_multitask_model,
    get_tokenizer,
    INTENT_LABELS,
    SENTIMENT_LABELS,
    NBA_LABELS,
)

MODEL_PATH = Path("models/multitask_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_tokenizer = None


def load_model_if_available():
    """Load multitask model if available."""
    global _model, _tokenizer
    _tokenizer = get_tokenizer()
    if MODEL_PATH.exists():
        _model = load_multitask_model(str(MODEL_PATH), device=DEVICE)
        print("Loaded trained multitask model.")
        return True
    else:
        _model = None
        print("No trained multitask model found yet.")
        return False


def get_model():
    """Get the loaded model and tokenizer."""
    return _model, _tokenizer


def is_model_loaded():
    """Check if model is loaded."""
    return _model is not None


def analyze_transcript(transcript_text: str):
    """
    Analyze transcript using multitask model.
    Returns dict with intent, sentiment, and nextBestAction.
    """
    if not is_model_loaded():
        raise ValueError("Model not trained yet. Call /train first after adding data.")

    enc = _tokenizer(
        transcript_text,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        intent_logits, sentiment_logits, nba_logits = _model(input_ids, attention_mask)

    intent_idx = torch.argmax(intent_logits, dim=-1).item()
    sentiment_idx = torch.argmax(sentiment_logits, dim=-1).item()
    nba_idx = torch.argmax(nba_logits, dim=-1).item()

    return {
        "intent": INTENT_LABELS[intent_idx],
        "sentiment": SENTIMENT_LABELS[sentiment_idx],
        "nextBestAction": NBA_LABELS[nba_idx],
    }

