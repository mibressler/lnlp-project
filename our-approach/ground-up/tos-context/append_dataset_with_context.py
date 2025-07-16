import csv
import os
import re
import requests

# OpenRouter API details
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-3.5-turbo"  # You can change this to another model available on OpenRouter
API_KEY = os.getenv('OPENROUTER_API_KEY')
if not API_KEY:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

def normalize(text):
    """Normalize text by removing extra spaces around punctuation and reducing multiple spaces."""
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Reduce multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def find_context_chunk(sentence, lines):
    """Find the first line containing the normalized sentence and return 20 lines before/after as a chunk."""
    if not sentence.strip():
        return None  # Empty sentence, no context
    
    sentence_norm = normalize(sentence)
    
    for i, line in enumerate(lines):
        line_norm = normalize(line)
        if sentence_norm in line_norm:
            # Found: grab 20 before and 20 after (inclusive of current line)
            start = max(0, i - 20)
            end = min(len(lines), i + 21)  # i + 20 + 1 for inclusive
            chunk = ''.join(lines[start:end])
            return chunk
    return None  # Not found

def get_llm_trimmed_context(chunk, sentence):
    """Call OpenRouter API to trim the chunk to a meaningful section."""
    prompt = f"""
Here is a chunk of text from a Terms of Service contract:

{chunk}

This chunk contains the following sentence: "{sentence}"

Please trim this chunk to the meaningful semantic section (such as a paragraph, heading, or section) that this sentence belongs to. Output only the trimmed text, nothing else.
"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"API error: {e}. Returning empty context.")
        return ""

def process_tsv(file_path, lines):
    """Process a TSV file, add 'context' column at index 6 (appended as the 7th column)."""
    rows = []
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = list(reader)
    
    if not rows:
        print(f"Empty file: {file_path}")
        return
    
    # Assume first row is header; append 'context' to it
    header = rows[0]
    if len(header) < 5:  # Need at least column 4
        raise ValueError(f"File {file_path} has fewer than 5 columns.")
    header.append('context')  # Append to end (becomes index len(header)-1, assumed to be 6)
    
    # Process data rows
    for row in rows[1:]:
        if len(row) < 5:
            print(f"Skipping invalid row in {file_path}: {row}")
            row.append('')  # Append empty context
            continue
        
        sentence = row[4]  # Column index 4
        chunk = find_context_chunk(sentence, lines)
        
        if chunk is None:
            context = ''
        else:
            context = get_llm_trimmed_context(chunk, sentence)
        
        row.append(context)  # Append to end
    
    # Write back to the same file (or change to a new file, e.g., file_path.replace('.tsv', '_with_context.tsv'))
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)  # Handle quoting for tabs/multilines
        writer.writerows(rows)
    
    print(f"Processed {file_path} successfully.")

def main():
    # Load the full documents once
    with open('all_tos_documents.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each file
    for file_path in ['test.tsv', 'train.tsv', 'val.tsv']:
        process_tsv(file_path, lines)

if __name__ == "__main__":
    main()