# BERT Baseline Classification Project

This project implements a baseline classification model using BERT to predict the "label" column in the provided dataset. The dataset is located in the `data` directory and is in TSV format.

## Project Structure

```
bert-baseline-classification
├── data
│   └── data.tsv          # Dataset containing features and labels
├── src
│   ├── data_preprocessing.py  # Handles data loading and preprocessing
│   ├── model_training.py      # Implements the BERT model and training
│   ├── evaluation.py          # Evaluates model performance
│   └── utils.py              # Contains utility functions
├── requirements.txt          # Lists project dependencies
├── README.md                 # Project documentation
└── .gitignore                # Files to ignore in version control
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd bert-baseline-classification
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Running the Project

1. **Data Preprocessing**:
   Run the data preprocessing script to prepare the dataset for training:
   ```
   python src/data_preprocessing.py
   ```

2. **Model Training**:
   Train the BERT model using the following command:
   ```
   python src/model_training.py
   ```

3. **Evaluation**:
   After training, evaluate the model's performance:
   ```
   python src/evaluation.py
   ```

## Results Interpretation

The evaluation script will output various metrics such as accuracy, precision, recall, and F1 score, which can be used to assess the model's performance on the classification task.

## License

This project is licensed under the MIT License - see the LICENSE file for details.