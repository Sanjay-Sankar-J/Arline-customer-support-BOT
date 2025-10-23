# backend/train_model.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

MODEL_DIR = r"C:\asap\intent_classifier_offline"
DATA_FILE = r"C:\asap\data\airline_intents_clean.csv"
FEEDBACK_FILE = r"C:\asap\feedback.csv"

# Load original dataset
orig_df = pd.read_csv(DATA_FILE)
unique_intents = orig_df['intent'].unique()
label2id = {label: idx for idx, label in enumerate(unique_intents)}
id2label = {idx: label for idx, label in enumerate(unique_intents)}
orig_df['labels'] = orig_df['intent'].map(label2id)

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Retraining function
def retrain_model(feedback_limit=1000, epochs=1):
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback to retrain.")
        return

    feedback_df = pd.read_csv(FEEDBACK_FILE, names=['text','intent'])
    if len(feedback_df) < feedback_limit:
        print(f"Feedback rows ({len(feedback_df)}) < {feedback_limit}. Skipping retrain.")
        return

    # Map feedback labels to existing IDs, add new labels if needed
    for label in feedback_df['intent'].unique():
        if label not in label2id:
            new_id = max(label2id.values()) + 1
            label2id[label] = new_id
            id2label[new_id] = label

    feedback_df['labels'] = feedback_df['intent'].map(label2id)

    # Combine with original dataset
    combined_df = pd.concat([orig_df, feedback_df.tail(feedback_limit)], ignore_index=True)

    # Update original dataset with feedback
    updated_orig_df = pd.concat([orig_df, feedback_df.tail(feedback_limit)], ignore_index=True)
    updated_orig_df.to_csv(DATA_FILE, index=False)
    print(f"Original dataset updated with {len(feedback_df.tail(feedback_limit))} new rows.")

    dataset = Dataset.from_pandas(combined_df)
    dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=64), batched=True)
    dataset.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    # Freeze base model if needed
    for param in model.distilbert.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir="./feedback_results",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()

    # Save updated model
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Model retrained and saved.")

    # Clear feedback
    open(FEEDBACK_FILE, 'w').close()
    print("Feedback CSV cleared after retrain.")

# Run retrain manually if needed
if __name__ == "__main__":
    retrain_model()
