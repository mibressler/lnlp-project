from typing import Optional

from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset

from utils.dataset import ClaudetteDataset
from utils.metrics import compute_binary_metrics, display_metrics


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(texts, padding=True, truncation=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def run(dataset: Optional[ClaudetteDataset] = None, *, epochs: int = 1) -> None:
    """Run a simple BERT fine-tuning baseline and print metrics."""
    dataset = dataset or ClaudetteDataset()
    train = dataset.get_dataset("train")

    texts = train["text"].tolist()
    labels = train["label"].tolist()
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_ds = BERTDataset(train_texts, train_labels, tokenizer)
    val_ds = BERTDataset(valid_texts, valid_labels, tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    args = TrainingArguments(
        "bert_out",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50,
        save_strategy="no",
        evaluation_strategy="epoch",
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    preds = trainer.predict(val_ds)
    binary_metrics = compute_binary_metrics(valid_labels, preds.predictions.argmax(axis=1))
    display_metrics("BERT baseline - Validation", binary_metrics)
