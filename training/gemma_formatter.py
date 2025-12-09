import json

def build_prompt(entry: dict):
    """
    Convert dataset entry into a Gemma training prompt
    """
    text = entry["text"]
    labels = entry["labels"]

    return (
        f"Analyze the following Hindi loan collection transcript.\n\n"
        f"Transcript:\n{text}\n\n"
        f"Provide analysis:\n"
        f"- Summary (short Hindi)\n"
        f"- Sentiment ({labels['sentiment']})\n"
        f"- Intent ({labels['intent']})\n"
        f"- Next Best Action ({labels['nba']})\n"
    )


def build_training_data(dataset: list):
    train_rows = []
    for entry in dataset:
        prompt = build_prompt(entry)
        train_rows.append({
            "instruction": prompt,
            "output": json.dumps(entry["labels"], ensure_ascii=False)
        })
    return train_rows
