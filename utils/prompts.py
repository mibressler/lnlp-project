import pandas as pd

def create_in_context_examples_prompt(input_output_pairs):
    base = """Here are some examples to guide your classification:"""
    
    def format_example(input_text, output_label):
        return f"\n**Input:** {input_text}\n**Output:** {output_label}\n"
    
    for input, output in input_output_pairs:
        base += "\n" + format_example(input, output)

    return base

def create_in_context_examples_prompt_auto(samples: pd.DataFrame): # use ClaudetteDataset.sample_rows_from_all_labels function's output
    from dataset.utils import ClaudetteDataset, INT_TO_CODE
    dataset = ClaudetteDataset()
    input_output_pairs = []
    for _, row in samples.iterrows():
        input_text = row['text']
        raw_output_label = row['label_indices']
        output_label = [INT_TO_CODE[label] for label in raw_output_label]
        input_output_pairs.append((input_text, output_label))
    
    return create_in_context_examples_prompt(input_output_pairs)


BINARY_TASK_INSTRUCTION = """You are an expert system designed to analyze clauses found in online terms of service documents, focusing on identifying terms that may be significantly disadvantageous to the consumer.

Your task is to perform a **binary classification** on the provided text, determining if it contains a clause that should be classified as **UNFAIR** or **FAIR**.

**Input:** You will receive a single sentence or a short text snippet extracted from an online terms of service document.

**Output:** You must provide **ONLY** one of the following two labels based on your analysis:
*   **UNFAIR**: If the input text contains a clause that falls under the criteria for being potentially or clearly disadvantageous to the consumer.
*   **FAIR**: If the input text contains a clause considered acceptable or beneficial to the consumer according to consumer protection principles, or if it does not contain a clause that meets the criteria for the UNFAIR category.

**Your output must contain nothing other than the single word label FAIR or UNFAIR.**
"""

DETAILED_BINARY_TASK_INSTRUCTION = """You are an expert system designed to analyze clauses found in online terms of service documents, focusing on identifying terms that may be significantly disadvantageous to the consumer.

Your task is to perform a **binary classification** on the provided text, determining if it contains a clause that should be classified as **UNFAIR** or **FAIR**.

**Input:** You will receive a single sentence or a short text snippet extracted from an online terms of service document.

**Output:** You must provide **ONLY** one of the following two labels based on your analysis:
*   **UNFAIR**: If the input text contains a clause that falls under the criteria for being potentially or clearly disadvantageous to the consumer.
*   **FAIR**: If the input text contains a clause considered acceptable or beneficial to the consumer according to consumer protection principles, or if it does not contain a clause that meets the criteria for the UNFAIR category.

**Criteria for Identifying an UNFAIR Clause:**
A clause is considered **UNFAIR** if it matches any of the following descriptions:

1.  **Jurisdiction:** The clause requires legal disputes to be handled by courts located in a place other than the consumer's country or place of residence.
2.  **Choice of Law:** The clause states that a law other than the law of the consumer's country or place of residence will govern the contract, potentially applying a foreign law.
3.  **Limitation of Liability:** The clause reduces, limits, or excludes the provider's responsibility for:
    *   Broad categories of losses (e.g., harm to computer systems, data loss, service unavailability).
    *   Losses described using general phrases like "to the fullest extent permissible by law".
    *   Serious issues such as physical injuries, intentional damages, or gross negligence.
4.  **Unilateral Change:** The clause permits the provider to modify or change the terms of service or the service itself unilaterally without requiring explicit agreement from the consumer.
5.  **Unilateral Termination:** The clause allows the provider to suspend or terminate the service or contract:
    *   Based on reasons specified by the provider.
    *   At any time, for any or no reason, and/or without providing prior notice to the consumer.
6.  **Contract by Using:** The clause states that the consumer automatically accepts the terms of service simply by using the service, without needing to perform an action like clicking "I agree".
7.  **Content Removal:** The clause gives the provider the right to modify or delete user content:
    *   Under certain specified conditions.
    *   At the provider's full discretion, at any time, for any or no reason, and/or without notice or the possibility for the user to retrieve the content.
8.  **Arbitration:** The clause requires or implies mandatory resolution of disputes through an arbitration process, especially if it specifies arbitration in a location other than the consumer's residence or suggests the process might not be based strictly on applicable law.

**Instructions:**
*   Analyze the provided text input carefully.
*   Determine if it contains a clause that fits any of the descriptions under the "Criteria for Identifying an UNFAIR Clause".
*   If it meets any of the criteria for an **UNFAIR** clause, output **ONLY** the label "**UNFAIR**".
*   If it does not meet any of the criteria for an **UNFAIR** clause (which includes clauses explicitly protecting consumer rights in these areas or text that is not a clause of these types), output **ONLY** the label "**FAIR**".
*   **Your output must contain nothing other than the single word label FAIR or UNFAIR.**
"""