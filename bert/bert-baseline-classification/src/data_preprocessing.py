import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    data = pd.read_csv(filepath, sep='\t')
    return data

def clean_data(data):
    # Example cleaning steps (customize as needed)
    data = data.dropna()  # Remove missing values
    return data

def preprocess_data(data):
    # Encode labels
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    
    # Split the dataset into features and labels
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == "__main__":
    data = load_data('data/data.tsv')
    cleaned_data = clean_data(data)
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(cleaned_data)