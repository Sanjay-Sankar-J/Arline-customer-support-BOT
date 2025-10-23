**Project Overview**

This repository contains a DistilBERT-based model for airline intent classification.
The model classifies customer queries into predefined intents and supports retraining using feedback data.
**
 Features**
 
Multi-class intent classification using DistilBERT.

Easy retraining with new feedback entries.

Dataset updates automatically after retraining.

Lightweight and efficient using DistilBertForSequenceClassification.

Tokenizer and model saved locally for offline usage.

**📂 Folder Structure**



distilbert_model/
 ├── config.json
 ├── model.safetensors
 ├── tokenizer.json
 ├── tokenizer_config.json
 ├── vocab.txt
 └── special_tokens_map.json

backend/
 └── train_model.py

data/
 └── airline_intents_clean.csv

feedback.csv  # Stores user feedback for retraining



**⚡ Requirements**

Python 3.9+

PyTorch

Transformers (pip install transformers)

Datasets (pip install datasets)

Pandas

Optional:

GPU support for faster retraining.

** Usage**
1. Load the model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "distilbert_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

2. Retrain the model with feedback
python backend/train_model.py


Feedback is stored in feedback.csv.

After retraining, the updated model overwrites the existing model in distilbert_model/.

feedback.csv is cleared automatically after retraining.

** Dataset**

Original dataset: data/airline_intents_clean.csv

Columns:

text → user query

intent → target label

Feedback dataset: feedback.csv

Columns: text, intent

**Notes**

Large model files (model.safetensors) are not ideal for GitHub; consider using Git LFS.

You can add new intents; retraining will automatically update label2id mapping.

📖 References

Hugging Face Transformers

DistilBERT Paper
