import sys, os
sys.path.append(os.getcwd())
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
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
OUTPUT_DIR = Path("./models/gemma_ft")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def format_for_training(example):
    """
    Format example for causal LM training.
    Combine instruction and output, tokenize, and create labels.
    Labels are masked for instruction part (-100), only output tokens contribute to loss.
    """
    # Combine instruction and output
    instruction = example["instruction"]
    output = example["output"]
    
    # Tokenize instruction part separately to know where to mask
    instruction_tokens = tokenizer(
        instruction,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=False,
    )
    instruction_len = len(instruction_tokens["input_ids"])
    
    # Tokenize full text (instruction + output) for training
    full_text = instruction + "\n" + output
    tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=1024,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    
    # Create labels: same as input_ids, but mask out instruction part
    # -100 is ignored in CrossEntropyLoss, so loss only computed on output
    labels = tokens["input_ids"].copy()
    
    # Mask out instruction tokens (don't compute loss on them)
    for i in range(min(instruction_len, len(labels))):
        labels[i] = -100
    
    # Also mask out padding tokens if any
    if "attention_mask" in tokens:
        for i in range(len(labels)):
            if tokens["attention_mask"][i] == 0:
                labels[i] = -100
    
    # Ensure labels is a list (not tensor) for the data collator
    tokens["labels"] = list(labels) if not isinstance(labels, list) else labels
    return tokens


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
    
    # Set pad_token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    hf_dataset = Dataset.from_list(rows)
    print(f"Created dataset with {len(hf_dataset)} examples")
    
    print("Tokenizing dataset...")
    tokenized = hf_dataset.map(format_for_training, remove_columns=hf_dataset.column_names)
    print(f"Tokenized dataset: {len(tokenized)} examples")
    
    # Verify one example has the right structure
    if len(tokenized) > 0:
        sample = tokenized[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Sample input_ids length: {len(sample.get('input_ids', []))}")
        print(f"Sample labels length: {len(sample.get('labels', []))}")
        print(f"Sample labels (first 10): {sample.get('labels', [])[:10]}")

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    if torch.cuda.is_available():
        model.to("cuda")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        save_strategy="epoch",
        logging_steps=1,  # Log more frequently to see progress
        fp16=torch.cuda.is_available(),
        report_to=None,  # Disable wandb/tensorboard
        logging_first_step=True,
        dataloader_num_workers=0,  # Disable multiprocessing to avoid issues
        remove_unused_columns=False,  # Keep our labels
        max_steps=10,  # Limit steps for testing (remove this later)
    )

    # Use DataCollatorForSeq2Seq which preserves our custom labels
    # This is better than DataCollatorForLanguageModeling which would overwrite labels
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Training...")
    print(f"Dataset size: {len(tokenized)} examples")
    print(f"Training for {args.num_train_epochs} epochs")
    
    try:
        train_result = trainer.train()
        print(f"Training completed! Loss: {train_result.training_loss}")
        
        print(f"Saving model to {OUTPUT_DIR}...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer (trainer.save_model already saves the model)
        trainer.save_model(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR), safe_serialization=True)
        
        # Verify files were saved
        model_files = list(OUTPUT_DIR.glob("*.safetensors")) + list(OUTPUT_DIR.glob("*.bin"))
        config_files = list(OUTPUT_DIR.glob("config.json"))
        tokenizer_files = list(OUTPUT_DIR.glob("tokenizer*.json")) + list(OUTPUT_DIR.glob("tokenizer_config.json"))
        
        print(f"Saved {len(model_files)} model file(s), {len(config_files)} config file(s), {len(tokenizer_files)} tokenizer file(s)")
        if model_files:
            print(f"Model files: {[f.name for f in model_files[:3]]}")  # Show first 3
        
        print("Gemma training finished successfully!")
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise
