# api/services/gemma_service.py
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = Path("models/gemma_ft")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_tokenizer = None


def is_model_loaded():
    """Check if Gemma model is loaded."""
    return _model is not None


def load_model():
    """Load Gemma model if available."""
    global _model, _tokenizer
    
    if not MODEL_PATH.exists():
        return False

    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    _model = AutoModelForCausalLM.from_pretrained(str(MODEL_PATH))
    
    if torch.cuda.is_available():
        _model.to(DEVICE)
    
    _model.eval()
    print("Loaded Gemma fine-tuned model.")
    return True


def analyze_transcript(transcript_text: str):
    """
    Analyze transcript using Gemma model.
    Returns JSON response with analysis.
    """
    if not is_model_loaded():
        # Try to load model if not loaded
        if not load_model():
            raise ValueError("Gemma model not trained yet! Hit /gemma/train first.")

    prompt = (
        f"Analyze Hindi loan collection call.\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Give JSON response with:\n"
        f"summary, sentiment, intent, nextBestAction\n"
    )

    inputs = _tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True
        )

    text = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Attempt to extract JSON
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        text = text[start:end]
        response = json.loads(text)
    except Exception:
        response = {"raw": text}

    return response

