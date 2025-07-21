import os

input_folder = os.path.join(os.path.dirname(__file__), 'all-documents-as-txt')
output_file = os.path.join(os.path.dirname(__file__), 'merged_tos_documents.txt')

with open(output_file, 'w', encoding='utf-8') as outfile:
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write('\n')