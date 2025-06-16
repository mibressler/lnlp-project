import random
import pandas as pd
import json
import re
import csv # Added for logging
from pathlib import Path
from datetime import datetime # Added for timestamping
import torch # Add this import

from utils.llm import get_llm_task_response
from utils.prompts import create_in_context_examples_prompt_auto, BINARY_TASK_INSTRUCTION
from utils.dataset import ClaudetteDataset, get_binary_labels
from utils.text import extract_json_from_text
from supar import Parser
from sklearn.metrics import f1_score

# Add this monkey patch to fix the torch.load issue
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

PARAPHRASE_SYSTEM_PROMPT = """You are a paraphrasing assistant. Your task is to rewrite the provided input text using different words and sentence structures while preserving the original meaning. Return the result in the following JSON format, and include nothing else in the output:
{
  "paraphrased_text": "..."
}
"""

dataset = None
constituency_parser = None
iterations = 5
candidates_per_iteration = 2
few_shot_samples = []
eval_data = []


def call_llm_for_scoring(prompt_text, instruction_text):
    print(f"      [LLM Call] Calling LLM for scoring. Instruction: '{instruction_text[:50]}...'")
    response = get_llm_task_response(
            user_prompt=prompt_text,
            system_prompt=instruction_text,
            few_shot_examples=create_in_context_examples_prompt_auto(few_shot_samples)
        ).content
    print(f"      [LLM Call] Received scoring response: '{response[:50]}...'")
    return response

def call_llm_for_paraphrase(phrase_to_paraphrase):
    print(f"    [LLM Call] Calling LLM for paraphrasing phrase: '{phrase_to_paraphrase}'")
    response = get_llm_task_response(
        user_prompt="phrase_to_paraphrase: " + phrase_to_paraphrase,
        system_prompt=PARAPHRASE_SYSTEM_PROMPT
    ).content
    print(f"    [LLM Call] Received paraphrase response: '{response[:100]}...'")

    extracted_json = extract_json_from_text(response)
    if extracted_json is None:
        print(f"    [LLM Call] Paraphrase parsing failed. Falling back to original phrase: '{phrase_to_paraphrase}'")
        return phrase_to_paraphrase

    paraphrased_text = extracted_json.get('paraphrased_text')
    print(f"    [LLM Call] Extracted paraphrased text: '{paraphrased_text}'")
    return paraphrased_text

# --- 2. Phrase Splitting ---

def extract_phrases_from_tree(tree):
    """
    Recursively extract phrases from a constituency parse tree.
    Returns a list of phrases (as strings).
    """
    phrases = []
    # If the node is a subtree with label (not a leaf)
    if hasattr(tree, 'label') and hasattr(tree, 'children'):
        # For phrase-level nodes (e.g., NP, VP, PP, etc.), collect the phrase
        if tree.label() in {"NP", "VP", "PP", "ADJP", "ADVP", "SBAR", "PRT"}:
            phrase = " ".join(tree.leaves())
            phrases.append(phrase)
        # Recurse into children
        for child in tree.children:
            phrases.extend(extract_phrases_from_tree(child))
    return phrases

def simple_phrase_splitter(text):
    print(f"  [Splitter] Attempting to split text into phrases: '{text}'")
    try:
        # Parse the sentence
        print(f"    [Splitter] Using constituency parser...")
        result = constituency_parser.predict([text], lang='en', prob=True, verbose=False)
        # supar returns a list of Sentence objects, each with .trees (list of Tree objects)
        trees = result.sentences[0].trees
        if trees:
            phrases = []
            for tree in trees:
                phrases.extend(extract_phrases_from_tree(tree))
            # Remove duplicates and empty strings
            phrases = [p.strip() for p in set(phrases) if p.strip()]
            if phrases:
                print(f"    [Splitter] Successfully split by parser: {phrases}")
                return phrases
    except Exception as e:
        print(f"    [Splitter] Constituency parser failed: {e}. Falling back to naive splitting.")
        # If parsing fails, fallback to naive splitting
        pass
    # Fallback: naive splitting
    print(f"    [Splitter] Using naive punctuation-based splitting.")
    parts = re.split(r'[.,;!?:]', text)
    phrases = [p.strip() for p in parts if p.strip()]
    if not phrases and text.strip():
        phrases = [word for word in text.split() if word.strip()]
        print(f"    [Splitter] Naive splitting by words: {phrases}")
    final_phrases = phrases if phrases else [text]
    print(f"  [Splitter] Final phrases: {final_phrases}")
    return final_phrases

# --- 3. Edit Operations ---
def delete_phrase_from_list(instruction_phrases, deleted_phrases_history):
    print(f"    [Edit Op] Attempting to delete from: {instruction_phrases}")
    if not instruction_phrases or len(instruction_phrases) <= 1: # Don't delete if only one phrase left
        print(f"    [Edit Op] Delete skipped (not enough phrases).")
        return list(instruction_phrases)
    phrases_copy = list(instruction_phrases)
    del_idx = random.randrange(len(phrases_copy))
    deleted_phrase = phrases_copy.pop(del_idx)
    deleted_phrases_history.append(deleted_phrase)
    print(f"    [Edit Op] Deleted phrase '{deleted_phrase}'. Remaining: {phrases_copy}. History: {deleted_phrases_history}")
    return phrases_copy

def add_phrase_to_list(instruction_phrases, deleted_phrases_history):
    print(f"    [Edit Op] Attempting to add to: {instruction_phrases} from history: {deleted_phrases_history}")
    if not deleted_phrases_history: # If no phrases have been deleted yet, do nothing
        print(f"    [Edit Op] Add skipped (no deleted phrases in history).")
        return list(instruction_phrases)
    phrases_copy = list(instruction_phrases)
    phrase_to_add = random.choice(deleted_phrases_history)
    add_idx = random.randrange(len(phrases_copy) + 1) # Can add at beginning, middle, or end
    phrases_copy.insert(add_idx, phrase_to_add)
    print(f"    [Edit Op] Added phrase '{phrase_to_add}'. Result: {phrases_copy}")
    return phrases_copy

def swap_phrases_in_list(instruction_phrases):
    print(f"    [Edit Op] Attempting to swap in: {instruction_phrases}")
    if not instruction_phrases or len(instruction_phrases) < 2: # Need at least two phrases to swap
        print(f"    [Edit Op] Swap skipped (not enough phrases).")
        return list(instruction_phrases)
    phrases_copy = list(instruction_phrases)
    idx1, idx2 = random.sample(range(len(phrases_copy)), 2)
    phrases_copy[idx1], phrases_copy[idx2] = phrases_copy[idx2], phrases_copy[idx1]
    print(f"    [Edit Op] Swapped phrases. Result: {phrases_copy}")
    return phrases_copy

# --- 4. Scoring/Evaluation ---
def evaluate_instruction(instruction_text, eval_data, task_labels):
    print(f"    [Evaluate] Evaluating instruction: '{instruction_text[:100]}...' on {len(eval_data)} items.")
    y_true = []
    y_pred = []
    if not eval_data:
        print("    [Evaluate] Evaluation data is empty. Returning 0.0, 0.0.")
        return 0.0, 0.0

    for i, (query_input_text, true_label) in enumerate(eval_data):
        if i < 3 or i % 10 == 0 : # Log first few and then every 10th
             print(f"      [Evaluate] Processing item {i+1}/{len(eval_data)}: Input: '{query_input_text[:50]}...'")
        prompt = f"Instruction: {instruction_text}\\n\\n"
        prompt += f"Now, for the following input, provide the output.\nInput: {query_input_text}\nOutput:"
        predicted_label = call_llm_for_scoring(prompt, instruction_text)
        y_true.append(true_label)
        y_pred.append(predicted_label)

    accuracy = sum([yt == yp for yt, yp in zip(y_true, y_pred)]) / len(y_true) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, labels=task_labels, average='macro', zero_division=0)
    print(f"    [Evaluate] Evaluation finished. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    return accuracy, f1

# --- 5. Main Search Loop ---
def main(num_iterations=5, num_candidates_per_iteration=2):
    print("[Main] Starting GRIPS search.")
    # --- Log File Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"paper_implementations/grips/grips_log_{timestamp}.csv"
    print(f"[Main] Log file will be: {log_filename}")
    log_fieldnames = ["timestamp", "iteration", "candidate_number", "edit_type", "candidate_instruction", "accuracy", "f1_score"]

    with open(log_filename, 'w', newline='') as csvfile:
        print(f"[Main] Opened log file: {log_filename}")
        writer = csv.DictWriter(csvfile, fieldnames=log_fieldnames)
        writer.writeheader()

        if not eval_data:
            print("[Main] Error: Evaluation data is empty. Please provide data. Exiting.")
            return

        # Infer task_labels from data (or you can define them explicitly)
        task_labels = sorted(list(set(label for _, label in eval_data)))
        if not task_labels:
            print("[Main] Error: Could not infer task labels from data. Ensure data has labels. Exiting.")
            return
        print(f"[Main] Inferred task labels: {task_labels}")

        # History of deleted phrases for the 'add' operation
        deleted_phrases_history = []
        print("[Main] Initialized empty deleted_phrases_history.")

        # --- Initialize Search ---
        print("[Main] Initializing search...")
        current_best_instruction_text = BINARY_TASK_INSTRUCTION
        initial_accuracy, initial_f1 = evaluate_instruction(current_best_instruction_text, eval_data, task_labels)
        current_best_score = initial_f1 # Using F1 as the primary score
        print(f"\\n[Main] Initial instruction: '{current_best_instruction_text}'")
        print(f"[Main] Initial score (acc, f1): ({initial_accuracy:.4f}, {initial_f1:.4f})")
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": 0,
            "candidate_number": 0,
            "edit_type": "initial",
            "candidate_instruction": current_best_instruction_text,
            "accuracy": initial_accuracy,
            "f1_score": initial_f1
        })

        # --- Search Loop ---
        for i in range(num_iterations):
            print(f"\\n--- [Main] Iteration {i+1}/{num_iterations} ---")
            
            current_instruction_phrases = simple_phrase_splitter(current_best_instruction_text)
            print(f"  [Main] Current instruction split into phrases: {current_instruction_phrases}")
            if not current_instruction_phrases:
                print(f"  [Main] Warning: Could not split current instruction into phrases: '{current_best_instruction_text}'. Using it as a single phrase.")
                current_instruction_phrases = [current_best_instruction_text]

            candidates_this_iteration = {} # Store as {instruction_text: score}

            for cand_num in range(num_candidates_per_iteration):
                print(f"  [Main] Generating Candidate {cand_num+1}/{num_candidates_per_iteration} for iteration {i+1}")
                
                _initial_operation_choice = random.choice(["delete", "add", "paraphrase", "swap"])
                operation_choice = _initial_operation_choice # operation_choice can be modified by re-sampling
                print(f"    [Main] Initial chosen edit operation: {operation_choice}")

                # If 'add' is chosen but not applicable (empty history), attempt to re-sample.
                if operation_choice == "add" and not deleted_phrases_history:
                    print(f"    [Main] 'Add' operation chosen but not applicable (history empty). Attempting to re-sample from other applicable operations.")
                    
                    applicable_alternatives = []
                    # Check applicability for other operations
                    if len(current_instruction_phrases) > 1: # For "delete" and "swap"
                        applicable_alternatives.extend(["delete", "swap"])
                    if current_instruction_phrases: # For "paraphrase"
                        applicable_alternatives.append("paraphrase")
                    
                    if applicable_alternatives:
                        operation_choice = random.choice(applicable_alternatives)
                        print(f"    [Main] Re-sampled due to non-applicable 'add'. New chosen edit operation: {operation_choice}")
                    else:
                        print(f"    [Main] Re-sampling for 'add' failed as no other operations are currently applicable. Original choice '{_initial_operation_choice}' will proceed; likely to hit fallback.")
                        # operation_choice remains _initial_operation_choice (i.e., "add")
                        # This will then likely fail the "add" condition below and go to the 'else' (fallback)

                edited_phrases = list(current_instruction_phrases) # Start with a copy for this candidate
                edit_type = "none" # Default, updated by successful operations or fallback

                # Apply chosen operation (original or re-sampled)
                if operation_choice == "delete" and len(current_instruction_phrases) > 1:
                    edited_phrases = delete_phrase_from_list(current_instruction_phrases, deleted_phrases_history)
                    edit_type = "delete"
                elif operation_choice == "add" and deleted_phrases_history:
                    edited_phrases = add_phrase_to_list(current_instruction_phrases, deleted_phrases_history)
                    edit_type = "add"
                elif operation_choice == "paraphrase" and current_instruction_phrases:
                    # Paraphrase a random phrase
                    phrase_to_paraphrase_idx = random.randrange(len(edited_phrases))
                    original_phrase = edited_phrases[phrase_to_paraphrase_idx]
                    print(f"    [Main] Paraphrasing phrase at index {phrase_to_paraphrase_idx}: '{original_phrase}'")
                    paraphrased_phrase = call_llm_for_paraphrase(original_phrase)
                    edited_phrases[phrase_to_paraphrase_idx] = paraphrased_phrase
                    edit_type = "paraphrase"
                elif operation_choice == "swap" and len(current_instruction_phrases) > 1:
                    edited_phrases = swap_phrases_in_list(current_instruction_phrases)
                    edit_type = "swap"
                else: # Fallback if chosen operation is not applicable by its conditions, or if 'add' re-sampling led here
                    print(f"    [Main] Chosen operation '{operation_choice}' (was initially '{_initial_operation_choice}') not applicable based on current state, or re-sampling led here. Attempting general fallback.")
                    if len(current_instruction_phrases) > 1 and deleted_phrases_history: # try add/delete if possible
                        if random.choice([True, False]):
                            print(f"      [Main Fallback] Trying delete operation.")
                            edited_phrases = delete_phrase_from_list(current_instruction_phrases, deleted_phrases_history)
                            edit_type = "delete (fallback)"
                        else:
                            print(f"      [Main Fallback] Trying add operation.")
                            edited_phrases = add_phrase_to_list(current_instruction_phrases, deleted_phrases_history)
                            edit_type = "add (fallback)"
                    elif current_instruction_phrases: # if add/delete not possible, try paraphrase
                        print(f"      [Main Fallback] Trying paraphrase operation.")
                        phrase_to_paraphrase_idx = random.randrange(len(edited_phrases))
                        original_phrase = edited_phrases[phrase_to_paraphrase_idx]
                        print(f"      [Main Fallback] Paraphrasing phrase at index {phrase_to_paraphrase_idx}: '{original_phrase}'")
                        paraphrased_phrase = call_llm_for_paraphrase(original_phrase)
                        edited_phrases[phrase_to_paraphrase_idx] = paraphrased_phrase
                        edit_type = "paraphrase (fallback)"
                    else: # if nothing else, keep as is
                        print(f"      [Main Fallback] No fallback operation applicable. Keeping phrases as is.")
                        edit_type = "none"
                
                candidate_instruction_text = " ".join(edited_phrases).strip()
                print(f"    [Main] Candidate instruction after edit '{edit_type}': '{candidate_instruction_text}'")
                
                # Basic cleanup: ensure not empty, capitalize first letter
                if not candidate_instruction_text:
                    print(f"    [Main] Candidate is empty. Falling back to BINARY_TASK_INSTRUCTION.")
                    candidate_instruction_text = BINARY_TASK_INSTRUCTION # Fallback
                else:
                    new_candidate_instruction_text = candidate_instruction_text[0].upper() + candidate_instruction_text[1:] if len(candidate_instruction_text) > 0 else ""
                    if new_candidate_instruction_text != candidate_instruction_text:
                        print(f"    [Main] Capitalized first letter. From: '{candidate_instruction_text}' To: '{new_candidate_instruction_text}'")
                        candidate_instruction_text = new_candidate_instruction_text


                if not candidate_instruction_text: # If it became empty after edits
                    print(f"    [Main] Candidate {cand_num+1} (edit: {edit_type}) resulted in empty string after cleanup, skipping.")
                    continue

                print(f"    [Main] Evaluating Candidate {cand_num+1} (edit: {edit_type}): '{candidate_instruction_text}'")
                accuracy, f1 = evaluate_instruction(candidate_instruction_text, eval_data, task_labels)
                # Using F1 as the primary score for optimization, but logging both
                print(f"    [Main] Candidate Score (acc, f1): ({accuracy:.4f}, {f1:.4f})")
                candidates_this_iteration[candidate_instruction_text] = f1 # Store F1 as the score for comparison

                # Log candidate
                writer.writerow({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "iteration": i + 1,
                    "candidate_number": cand_num + 1,
                    "edit_type": edit_type,
                    "candidate_instruction": candidate_instruction_text,
                    "accuracy": accuracy,
                    "f1_score": f1
                })
            
            # Select the best candidate from this iteration
            if candidates_this_iteration:
                # item[1] is the F1 score
                iteration_best_instruction, iteration_best_score_f1 = max(candidates_this_iteration.items(), key=lambda item: item[1])
                
                if iteration_best_score_f1 > current_best_score: # current_best_score is F1
                    print(f"  [Main] New best instruction found in iteration {i+1}: '{iteration_best_instruction}' (F1 Score: {iteration_best_score_f1:.4f})")
                    current_best_instruction_text = iteration_best_instruction
                    current_best_score = iteration_best_score_f1
                else:
                    print(f"  [Main] No improvement in iteration {i+1}. Best F1 score remains {current_best_score:.4f} with instruction: '{current_best_instruction_text}'")
            else:
                print(f"  [Main] No valid candidates generated in iteration {i+1}.")

        # --- Output Final Result ---
        print(f"\\n--- [Main] Search Finished ---")
        print(f"[Main] Best instruction found: '{current_best_instruction_text}'")
        print(f"[Main] Best F1 score: {current_best_score:.4f}")
        print(f"[Main] Log file created: {log_filename}")
    print("[Main] GRIPS search ended.")



def run(
    ds: Optional[ClaudetteDataset] = None,
    *,
    iterations: int = 5,
    candidates_per_iteration: int = 2,
    sample_size: int = 20,
) -> None:
    """Execute the GRIPS baseline search with minimal setup."""
    global dataset, constituency_parser, eval_data, few_shot_samples
    Path("paper_implementations/grips").mkdir(parents=True, exist_ok=True)
    dataset = ds or ClaudetteDataset()
    constituency_parser = Parser.load("crf-con-en")
    few_shot_samples = pd.concat([
        dataset.sample_rows_from_all_unfair_labels("train", 1),
        dataset.fetch_rows_by_label("train", 0).sample(9),
    ])
    eval_df = get_binary_labels(dataset.get_dataset("test")).sample(sample_size, random_state=42)
    eval_data = list(eval_df.itertuples(index=False, name=None))
    main(iterations, candidates_per_iteration)

