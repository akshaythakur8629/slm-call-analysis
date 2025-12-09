# api/services/dataset_service.py
import json
from pathlib import Path
from ..labeler import (
    detect_nba_rules,
    detect_sentiment_rules,
    detect_intent_rules,
)

DATASET_PATH = Path("data/training_dataset.jsonl")
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_transcript_to_dataset(transcript_json: dict):
    """
    Convert interaction_transcript JSON into a single text block
    and append to dataset with auto-labels.
    """
    formatted_text = ""

    for t in transcript_json["interaction_transcript"]:
        role = t.get("role", "user").upper()
        text = t.get("en_text", "")
        formatted_text += f"{role}: {text}\n"

    full_text = formatted_text.strip()

    nba = detect_nba_rules(full_text)
    sentiment = detect_sentiment_rules(full_text)
    intent = detect_intent_rules(full_text)

    entry = {
        "text": full_text,
        "labels": {
            "intent": intent,
            "sentiment": sentiment,
            "nba": nba,
        },
    }

    with DATASET_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"status": "stored", "labels": entry["labels"]}


def load_training_dataset():
    if not DATASET_PATH.exists():
        return []

    data = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

