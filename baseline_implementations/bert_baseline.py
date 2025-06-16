from typing import Optional
import sys
import os

# Add the parent directory to the Python path so we can import from utils and config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

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
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def run(
    dataset: Optional[ClaudetteDataset] = None,
    *,
    epochs: int = 1,
    sample_size: int = 200,
):
    """Run a simple BERT fine-tuning baseline and return metrics."""
    dataset = dataset or ClaudetteDataset()
    train = dataset.get_dataset("train")
    test_sample = dataset.get_dataset("test").sample(sample_size, random_state=42)

    texts = train["text"].tolist()
    labels = train["label"].tolist()
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_ds = BERTDataset(train_texts, train_labels, tokenizer)
    val_ds = BERTDataset(valid_texts, valid_labels, tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=2
    )
    
    # Training arguments for transformers 4.52.4 with CPU usage
    args = TrainingArguments(
        output_dir="bert_out",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
        use_cpu=True,
        dataloader_num_workers=0,
        report_to=[],  # Empty list instead of None
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    
    print("Starting BERT training (CPU-only)...")
    trainer.train()
    print("Training completed!")

    print("Running evaluation...")
    preds = trainer.predict(val_ds)
    predictions = preds.predictions.argmax(axis=1)

    binary_val_metrics = compute_binary_metrics(valid_labels, predictions)
    display_metrics("BERT baseline - Validation", binary_val_metrics)

    # Evaluate on test sample
    test_texts = test_sample["text"].tolist()
    test_labels = test_sample["label"].tolist()
    test_ds = BERTDataset(test_texts, test_labels, tokenizer)
    test_preds = trainer.predict(test_ds)
    test_predictions = test_preds.predictions.argmax(axis=1)
    binary_test_metrics = compute_binary_metrics(test_labels, test_predictions)
    display_metrics("BERT baseline - Test", binary_test_metrics)

    params = f"model=bert-base-uncased epochs={epochs} device=cpu"
    sample_n = len(test_sample)

    return {
        "binary_val": binary_val_metrics,
        "binary_test": binary_test_metrics,
        "multi_val": None,
        "multi_test": None,
        "params": params,
        "sample_size": sample_n,
    }


if __name__ == "__main__":
    print("Running BERT baseline test...")
    metrics = run(epochs=1)
    print("Test completed!")
    print(f"Metrics: {metrics}")