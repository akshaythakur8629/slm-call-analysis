from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

ENCODER_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

INTENT_LABELS = [
    "WILLING_TO_PAY",
    "NEEDS_EXTENSION",
    "EMI_REDUCTION_REQUEST",
    "DISPUTE_OR_COMPLAINT",
    "NOT_INTERESTED",
    "UNKNOWN_INTENT",
]

SENTIMENT_LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

NBA_LABELS = [
    "PTP",
    "EXTENSION",
    "EMI_RESTRUCTURE",
    "COMPLAINT_SUPPORT",
    "LEGAL_WARNING",
    "NO_RESPONSE",
    "CUSTOMER_NOT_SURE",
]


class MultiTaskClassifier(nn.Module):
    def __init__(self, encoder_name: str = ENCODER_MODEL_NAME, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.intent_head = nn.Linear(hidden_size, len(INTENT_LABELS))
        self.sentiment_head = nn.Linear(hidden_size, len(SENTIMENT_LABELS))
        self.nba_head = nn.Linear(hidden_size, len(NBA_LABELS))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token / pooled output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]  # CLS

        intent_logits = self.intent_head(pooled)
        sentiment_logits = self.sentiment_head(pooled)
        nba_logits = self.nba_head(pooled)

        return intent_logits, sentiment_logits, nba_logits


def get_tokenizer():
    return AutoTokenizer.from_pretrained(ENCODER_MODEL_NAME)


def save_multitask_model(model: MultiTaskClassifier, path: str):
    to_save = {
        "state_dict": model.state_dict(),
    }
    torch.save(to_save, path)


def load_multitask_model(path: str, device: str = "cpu") -> MultiTaskClassifier:
    model = MultiTaskClassifier()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model
