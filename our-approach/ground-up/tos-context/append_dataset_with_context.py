import csv
import os
import re
import requests
import difflib
from dotenv import load_dotenv  # Ensure python-dotenv is installed: pip install python-dotenv

# Dynamically find the repo root by walking up until we find .env
def find_repo_root(start_dir):
    current_dir = start_dir
    while current_dir != os.path.dirname(current_dir):  # Stop at drive root
        if os.path.exists(os.path.join(current_dir, '.env')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None  # Not found

# Get script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] Script directory: {SCRIPT_DIR}")

# Find repo root and load .env if found
repo_root = find_repo_root(SCRIPT_DIR)
if repo_root:
    print(f"[DEBUG] Repo root found: {repo_root}")
    load_dotenv(dotenv_path=os.path.join(repo_root, '.env'))
    print("[DEBUG] Loaded .env from repo root.")
else:
    print("[DEBUG] Warning: .env not found in repo root hierarchy. Falling back to environment variables.")

# OpenRouter API details
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.0-flash-001"  # You can change this to another model available on OpenRouter
API_KEY = os.getenv('OPENROUTER_API_KEY')
if API_KEY:
    print("[DEBUG] OPENROUTER_API_KEY loaded successfully (length: {len(API_KEY)}).")
else:
    raise ValueError("OPENROUTER_API_KEY not found in environment or .env file.")

def extract_words(text):
    """Extract lowercase words, ignoring punctuation, extra spaces, and handling contractions."""
    # Remove punctuation except apostrophes in contractions
    text = re.sub(r"[^\w\s']", '', text)
    # Split into words
    words = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    return [word for word in words if word]

def find_context_chunk(sentence, lines, words_with_lines):
    """Find the first fuzzy occurrence of the sentence's word sequence in the document and return 20 lines before/after as a chunk."""
    if not sentence.strip():
        return None  # Empty sentence, no context
    
    sentence_words = extract_words(sentence)
    if not sentence_words:
        return None
    
    len_s = len(sentence_words)
    total_words = len(words_with_lines)
    threshold = 0.85  # Adjustable similarity threshold (1.0 = exact match)
    
    for start in range(total_words - len_s + 1):
        candidate_words = [w for _, w in words_with_lines[start:start + len_s]]
        matcher = difflib.SequenceMatcher(None, sentence_words, candidate_words)
        ratio = matcher.ratio()
        if ratio >= threshold:
            # Found match! Get the line indices spanned by this match
            line_idxs = [line_idx for line_idx, _ in words_with_lines[start:start + len_s]]
            min_line = min(line_idxs)
            max_line = max(line_idxs)
            
            # Grab 20 lines before min_line and 20 after max_line (inclusive)
            chunk_start = max(0, min_line - 10)
            chunk_end = min(len(lines), max_line + 11)  # +21 to include max_line + 20 after
            chunk = ''.join(lines[chunk_start:chunk_end])
            print(f"[DEBUG]   Fuzzy match found with ratio {ratio:.2f} spanning lines {min_line}-{max_line}.")
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
        #print("[DEBUG] Calling OpenRouter API for trimming...")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        trimmed = result['choices'][0]['message']['content'].strip()
        #print(f"[DEBUG] LLM response received (trimmed length: {len(trimmed)}).")
        return trimmed
    except Exception as e:
        print(f"[DEBUG] API error: {e}. Returning empty context.")
        return ""

def process_tsv(file_name, lines, words_with_lines):
    """Process a TSV file, add 'context' column at index 6 (appended as the 7th column)."""
    file_path = os.path.join(SCRIPT_DIR, file_name)
    print(f"[DEBUG] Processing TSV: {file_path}")
    
    rows = []
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        rows = list(reader)
    
    if not rows:
        print(f"[DEBUG] Empty file: {file_path}")
        return
    
    # Assume first row is header; append 'context' to it
    header = rows[0]
    print(f"[DEBUG] Header: {header} (columns: {len(header)})")
    if len(header) < 5:  # Need at least column 4
        raise ValueError(f"File {file_path} has fewer than 5 columns.")
    header.append('context')  # Append to end (becomes index len(header)-1, assumed to be 6)
    
    # Counters for summary
    total_rows = len(rows) - 1  # Exclude header
    contexts_added = 0
    empties = 0
    
    # Process data rows
    for idx, row in enumerate(rows[1:], start=1):
        if len(row) < 5:
            print(f"[DEBUG] Row {idx}: Skipping invalid row (too few columns): {row}")
            row.append('')  # Append empty context
            empties += 1
            continue
        
        sentence = row[4]  # Column index 4
        sent_preview = sentence[:100] + '...' if len(sentence) > 100 else sentence
        print(f"[DEBUG] Row {idx}/{total_rows}: Processing sentence: '{sent_preview}'")
        
        chunk = find_context_chunk(sentence, lines, words_with_lines)
        
        if chunk is None:
            print("[DEBUG]   No chunk found (sentence not matched).")
            context = ''
            empties += 1
        else:
            #chunk_preview = chunk[:200] + '...' if len(chunk) > 200 else chunk
            #print(f"[DEBUG]   Chunk found (preview): '{chunk_preview}'")
            context = get_llm_trimmed_context(chunk, sentence)
            contexts_added += 1
        
        context_preview = context[:100] + '...' if len(context) > 1000 else context
        print(f"[DEBUG]   Final context: '{context_preview}'")
        row.append(context)  # Append to end
    
    # Write back to the same file
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)  # Handle quoting for tabs/multilines
        writer.writerows(rows)
    
    print(f"[DEBUG] Processed {file_path} successfully. Summary: {contexts_added} rows with context, {empties} empty (out of {total_rows} data rows).")

def main():
    # Load the full documents once (using script dir)
    tos_path = os.path.join(SCRIPT_DIR, 'all_tos_documents.txt')
    print(f"[DEBUG] Loading all_tos_documents.txt from: {tos_path}")
    with open(tos_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"[DEBUG] Loaded {len(lines)} lines from all_tos_documents.txt.")
    
    # Precompute words_with_lines for robust matching
    words_with_lines = []
    for line_idx, line in enumerate(lines):
        line_words = extract_words(line)
        for word in line_words:
            words_with_lines.append((line_idx, word))
    print(f"[DEBUG] Precomputed {len(words_with_lines)} words for matching.")
    
    # Process each file
    for file_name in ['test.tsv', 'train.tsv', 'val.tsv']:
        process_tsv(file_name, lines, words_with_lines)

if __name__ == "__main__":
    main()