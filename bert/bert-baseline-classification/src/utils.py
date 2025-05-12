import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

def load_data(filepath):
    if os.path.exists(filepath):
        data = pd.read_csv(filepath, sep='\t')
        return data
    else:
        raise FileNotFoundError(f"The file {filepath} does not exist.")

def tokenize_data(texts, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return tokens

def plot_training_history(history):
    plt.plot(history['loss'], label='Loss')
    plt.plot(history['accuracy'], label='Accuracy')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()

def save_model(model, path):
    model.save_pretrained(path)

def load_model(path):
    return BertForSequenceClassification.from_pretrained(path)