import csv
import os

def get_documents(filename):
    documents = set()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            if len(row) > 1:
                documents.add(row[1].strip())
    return documents

# Get unique documents from each file
test_docs = get_documents('test.tsv')
train_docs = get_documents('train.tsv')
val_docs = get_documents('val.tsv')

# Find union (all unique documents from any of the three files)
all_docs = test_docs | train_docs | val_docs

# Sort for neatness
all_list = sorted(all_docs)

# Write to file
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'list_of_documents.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for doc in all_list:
        f.write(doc + '\n')