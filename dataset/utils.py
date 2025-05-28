import pandas as pd
from typing import Dict
import os

# Mapping between string code and integer label, starting from 1 and setting 0 for "FAIR"
CODE_TO_INT = {
    "FAIR": 0,   # Fairness
    "A": 1,      # Arbitration
    "CH": 2,     # Unilateral change
    "CR": 3,     # Content removal
    "J": 4,      # Jurisdiction
    "LAW": 5,    # Choice of law
    "LTD": 6,    # Limitation of liability
    "PINC": 7,   # ?
    "TER": 8,    # Unilateral termination
    "USE": 9     # Contract by using
}

# Reverse mapping from integer label to string code
INT_TO_CODE = {v: k for k, v in CODE_TO_INT.items()}

# Optional: Full string descriptions for each code
CODE_TO_FULL = {
    "FAIR": "Fairness",
    "A": "Arbitration",
    "CH": "Unilateral change",
    "CR": "Content removal",
    "J": "Jurisdiction",
    "LAW": "Choice of law",
    "LTD": "Limitation of liability",
    "PINC": "?",
    "TER": "Unilateral termination",
    "USE": "Contract by using"
}

def get_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['output'] = df['label_indices'].apply(lambda indices: "FAIR" if indices == [0] else "UNFAIR")
    return df[['text', 'output']]

class ClaudetteDataset:
    def __init__(self):
        self.splits = ['train', 'val', 'test', 'all']
        self.datasets: Dict[str, pd.DataFrame] = {}

    def _load_and_preprocess(self, split: str) -> pd.DataFrame:
        # TODO: Update the file path to match your directory structure - SOLUTION HERE IS NOT UNIVERSAL - MIGHT FAIL
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(root_dir, f'dataset/claudette_{split}_merged.tsv')
        df = pd.read_csv(file_path, sep='\t')
        # Preprocess 'label_indices' column
        if df['label_indices'].dtype == object:
            df['label_indices'] = df['label_indices'].apply(
                lambda x: [int(i) for i in str(x).strip('[]').split(',') if i.strip().isdigit()]
            )
        return df

    def get_dataset(self, split: str) -> pd.DataFrame:
        assert split in self.splits, "Invalid split"
        if split not in self.datasets:
            self.datasets[split] = self._load_and_preprocess(split)
        return self.datasets[split]

    def fetch_rows_by_label(self, split: str, label: int) -> pd.DataFrame:
        # Example usage: dataset.fetch_rows_by_label('train', 3).sample(10, random_state=42)['text'].iloc[0]
        assert split in self.splits, "Invalid split"
        assert isinstance(label, int) and 0 <= label <= 9, "Label must be an integer between 0 and 9"
        df = self.get_dataset(split)
        filtered_df = df[df['label_indices'].apply(lambda indices: label in indices)]
        return filtered_df
    
    def sample_rows_from_all_unfair_labels(self, split: str, n: int = 2) -> pd.DataFrame:
        # Example usage: dataset.sample_rows_by_all_labels('train', 10)
        assert split in self.splits, "Invalid split"
        df = self.get_dataset(split)
        sampled_dfs = []
        for label in range(1, 10):
            filtered_df = self.fetch_rows_by_label(split, label).sample(n=min(n, len(df)))
            if not filtered_df.empty:
                sampled_df = filtered_df.sample(n=min(n, len(filtered_df)))
                sampled_dfs.append(sampled_df)

        return pd.concat(sampled_dfs, ignore_index=True) if sampled_dfs else pd.DataFrame()