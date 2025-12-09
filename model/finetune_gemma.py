import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.gemma_formatter import build_training_data
from api.services.dataset_service import load_training_dataset

BASE_MODEL = "google/gemma-2-2b"
OUTPUT_DIR = "./models/gemma_ft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_for_training(example):
    full_text = example["instruction"] + "\n" + example["output"]
    return tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
    )


if __name__ == "__main__":
    dataset = load_training_dataset()
    if not dataset:
        print("Dataset empty. Add transcripts first!")
        exit()

    print("Preparing training data...")
    rows = build_training_data(dataset)

    print("Loading model/tokenizer:", BASE_MODEL)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    hf_dataset = Dataset.from_list(rows)
    tokenized = hf_dataset.map(format_for_training)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    if torch.cuda.is_available():
        model.to("cuda")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        logging_steps=5,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
    )

    print("Training...")
    trainer.train()

    print("Saving model to", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Gemma training finished!")
