import csv
import os
import re
import requests
import difflib
import multiprocessing
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
MODEL = "google/gemini-2.5-flash-lite-preview-06-17"  # You can change this to another model available on OpenRouter
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
    threshold = 0.8  # Adjustable similarity threshold (lowered for more matches; 1.0 = exact)
    
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
            chunk_start = max(0, min_line - 20)
            chunk_end = min(len(lines), max_line + 21)  # +21 to include max_line + 20 after
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
    
    for attempt in range(2):  # Retry once on failure
        try:
            print("[DEBUG] Calling OpenRouter API for trimming... (attempt {attempt + 1})")
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            trimmed = result['choices'][0]['message']['content'].strip()
            print(f"[DEBUG] LLM response received (trimmed length: {len(trimmed)}).")
            return trimmed
        except Exception as e:
            print(f"[DEBUG] API error on attempt {attempt + 1}: {e}")
    print("[DEBUG] All API attempts failed. Returning empty context.")
    return ""

def process_row(row_data):
    """Process a single row (for parallel execution)."""
    idx, row, lines, words_with_lines, sentence_col, total_rows = row_data
    if len(row) < sentence_col + 1:
        print(f"[DEBUG] Row {idx}: Skipping invalid row (too few columns).")
        return row, ''
    
    sentence = row[sentence_col]
    sent_preview = sentence[:100] + '...' if len(sentence) > 100 else sentence
    print(f"[DEBUG] Row {idx}/{total_rows}: Processing sentence: '{sent_preview}'")
    
    chunk = find_context_chunk(sentence, lines, words_with_lines)
    
    if chunk is None:
        print("[DEBUG]   No chunk found (sentence not matched).")
        return row, ''
    else:
        chunk_preview = chunk[:200] + '...' if len(chunk) > 200 else chunk
        print(f"[DEBUG]   Chunk found (preview): '{chunk_preview}'")
        context = get_llm_trimmed_context(chunk, sentence)
        context_preview = context[:100] + '...' if len(context) > 100 else context
        print(f"[DEBUG]   Final context: '{context_preview}'")
        return row, context

def process_tsv(file_name, lines, words_with_lines, max_workers=None):
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
    sentence_col = 4  # Column index for sentence
    if len(header) < sentence_col + 1:
        raise ValueError(f"File {file_path} has fewer than {sentence_col + 1} columns.")
    
    # Caching: Skip if 'context' already exists
    if 'context' in header:
        print(f"[DEBUG] 'context' column already exists in {file_path}. Skipping processing.")
        return
    
    header.append('context')  # Append to end (becomes index len(header)-1, assumed to be 6)
    
    # Counters for summary
    total_rows = len(rows) - 1  # Exclude header
    contexts_added = 0
    empties = 0
    
    # Prepare data for parallel processing
    pool_data = [(idx + 1, row, lines, words_with_lines, sentence_col, total_rows) for idx, row in enumerate(rows[1:])]
    
    # Set max_workers conservatively to avoid API rate limits
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count() // 2, 4)  # e.g., 4 on an 8-core machine
    
    print(f"[DEBUG] Starting parallel processing with {max_workers} workers.")
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.map(process_row, pool_data)
    
    # Update rows with results
    for i, (updated_row, context) in enumerate(results):
        rows[i + 1] = updated_row  # Update the row (though it might not change)
        rows[i + 1].append(context)
        if context:
            contexts_added += 1
        else:
            empties += 1
    
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
    
    # Process each file with parallelism (adjust max_workers if needed)
    for file_name in ['test.tsv', 'train.tsv', 'val.tsv']:
        process_tsv(file_name, lines, words_with_lines, max_workers=4)

if __name__ == "__main__":
    main()