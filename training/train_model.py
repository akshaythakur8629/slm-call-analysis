import json
import sys
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.multitask_model import (
    MultiTaskClassifier,
    get_tokenizer,
    INTENT_LABELS,
    SENTIMENT_LABELS,
    NBA_LABELS,
    save_multitask_model,
)

DATA_PATH = Path("data/training_dataset.jsonl")
MODEL_OUT = Path("models/multitask_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CallDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_len: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        text = item["text"]
        labels = item["labels"]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        intent_idx = INTENT_LABELS.index(labels["intent"])
        sentiment_idx = SENTIMENT_LABELS.index(labels["sentiment"])
        nba_idx = NBA_LABELS.index(labels["nba"])

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "intent": torch.tensor(intent_idx, dtype=torch.long),
            "sentiment": torch.tensor(sentiment_idx, dtype=torch.long),
            "nba": torch.tensor(nba_idx, dtype=torch.long),
        }


def load_dataset() -> List[Dict]:
    rows = []
    if not DATA_PATH.exists():
        print("No dataset found at", DATA_PATH)
        return rows

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def train():
    rows = load_dataset()
    if not rows:
        print("No training data, exiting.")
        return

    tokenizer = get_tokenizer()
    dataset = CallDataset(rows, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = MultiTaskClassifier(freeze_encoder=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            intent_labels = batch["intent"].to(DEVICE)
            sentiment_labels = batch["sentiment"].to(DEVICE)
            nba_labels = batch["nba"].to(DEVICE)

            optimizer.zero_grad()
            intent_logits, sentiment_logits, nba_logits = model(input_ids, attention_mask)

            loss_intent = criterion(intent_logits, intent_labels)
            loss_sentiment = criterion(sentiment_logits, sentiment_labels)
            loss_nba = criterion(nba_logits, nba_labels)

            loss = loss_intent + loss_sentiment + loss_nba

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    save_multitask_model(model, str(MODEL_OUT))
    print("Saved model to", MODEL_OUT)


if __name__ == "__main__":
    train()
