from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch, json

MODEL_NAME = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="data/training_dataset.jsonl")["train"]

def format_example(example):
    text = (
        f"Instruction: {example['instruction']}\n"
        f"Input:\n{example['input']}\n\n"
        f"Output:\n{json.dumps(example['output'], ensure_ascii=False)}"
    )
    tokens = tokenizer(text, truncation=True, max_length=768)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(format_example, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_steps=2,
    save_strategy="epoch",
    learning_rate=5e-5,
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete. Model saved in ./model")
