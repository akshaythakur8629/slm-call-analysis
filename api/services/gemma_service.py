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

    try:
        # Try loading tokenizer - let transformers decide the best approach
        _tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: Failed to load tokenizer from {MODEL_PATH}: {e}")
        # If loading from saved path fails, try loading from base model
        # This can happen if tokenizer files weren't saved correctly
        try:
            BASE_MODEL = "google/gemma-2-2b"
            _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            print("Loaded tokenizer from base model instead.")
        except Exception as e2:
            print(f"Error: Failed to load tokenizer: {e2}")
            return False
    
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            trust_remote_code=True
        )
        
        if torch.cuda.is_available():
            _model.to(DEVICE)
        
        _model.eval()
        print("Loaded Gemma fine-tuned model.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


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

