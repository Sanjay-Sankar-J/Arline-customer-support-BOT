1.**Project Overview**

This repository contains a DistilBERT-based model for airline intent classification.
The model classifies customer queries into predefined intents and supports retraining using feedback data.
**
 Features**
 
Multi-class intent classification using DistilBERT.

Easy retraining with new feedback entries.

Dataset updates automatically after retraining.

Lightweight and efficient using DistilBertForSequenceClassification.

Tokenizer and model saved locally for offline usage.

2.**📂 Folder Structure**



<img width="407" height="402" alt="image" src="https://github.com/user-attachments/assets/6f8e11a8-4c96-40e1-b1d5-04ff3f45d495" />


feedback.csv  # Stores user feedback for retraining



3.**Requirements**

Python 3.9+

PyTorch

Transformers (pip install transformers)

Datasets (pip install datasets)

Pandas

Optional:

GPU support for faster retraining.

4.**Usage**

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

**Dataset**

Original dataset: data/airline_intents_clean.csv

Columns:

text → user query

intent → target label

Feedback dataset: feedback.csv

Columns: text, intent

5.**Notes**

Large model files (model.safetensors) are not ideal for GitHub; consider using Git LFS.

You can add new intents; retraining will automatically update label2id mapping.

6.**References**

Hugging Face Transformers

DistilBERT Paper
