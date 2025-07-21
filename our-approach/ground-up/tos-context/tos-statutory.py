import os
import random
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import openai
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import csv  # Import csv for proper reading

csv.field_size_limit(10_000_000) 

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# ========== Strategy Lists ============

instruction_strategies = [
    "Crafting an expert who is an expert at the given task, by writing a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective.",
    "Explaining step-by-step how the problem should be tackled, and making sure the model explains step-by-step how it came to the answer. You can do this by adding \"Let's think step-by-step\".",
    "Imagining three different experts who are discussing the problem at hand. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.",
    "Making sure all information needed is in the prompt, adding where necessary but making sure the question remains having the same objective.",
    "At the end of the prompt, add a phrase that evokes a strong emotion. When doing so, keep the following four points in mind:\n1. Define emotional goals: Identify the emotional response you want to evoke, such as encouragement, motivation, or reassurance.\n2. Use positive language: Incorporate words and phrases that are positive and supportive. Examples include \"believe in your abilities,\" \"excellent,\" \"success,\" and \"outstanding achievements\".\n3. Emphasize key words: Use techniques like exclamation marks and capitalized words to highlight important aspects and to enhance the emotional impact.\n4. Incorporate social and self-esteem cues: Design stimuli that leverage social influence (e.g., group membership, others' opinions) and boost self-esteem and motivation. This can help regulate the emotional response of the Large Language Models and tap into intrinsic motivation.",
    "For a given prompt, add a phrase such as \"Read the question again\" that instructs the Large Language Models to reread the question before generating an answer. This strategy is particularly effective for complex tasks and helps enhance the quality and reliability of the model's outputs",
    "Clearly define the desired style in the given prompt. For example, you might say, \"Write a formal letter about...\" or \"Create a casual conversation discussing...\". This guidance helps the model produce text that matches the requested stylistic elements, whether it's formal, informal, technical, or poetic.",
    "For a given prompt, add a phrase that instructs the Large Language Models to rephrase the question before responding, such as \"Rephrase and expand the question, and respond.\"",
    "Make the description of the given prompt more specific. This makes it easier for Large Language Models to correctly execute prompt instructions.",
    "To allow Large Language Models to make logical and unbiased inferences, add phrases to a given prompt that instruct it to remove opinionated content. This helps the model concentrate on providing responses based on careful analysis and logical reasoning, minimizing biases.",
    "If a given prompt has long instructions, make it shorter by condensing it to only the essential parts. Never remove the instruction to strictly respond with '0' for fair or '1' for unfair.",
    "Edit the prompt instruction to invoke legal reasoning for problem solving: 1) State the goal of determining the unfairness of a clause. 2) Give a detailed definition of what could be considered an unfair clause. 3) compare the given sentence with the definition to estimate which parts of the sentence falls under that definition. 4) make a final determination based on the comparison."
    "Craft a concise description of the most capable expert for the task, addressing them in second person (e.g., 'You are an expert in...') to enhance precision and focus the model's response.",
    "Guide the model through step-by-step reasoning by adding a precise phrase like 'Let's think step-by-step' at the end, ensuring explanations are logical and concise while shortening unnecessary elaboration.",
    "Envision three experts collaboratively solving the problem: each briefly shares one step of thinking per round, and any who realize they're wrong exit immediately. This promotes precise, error-minimizing reasoning without verbose discussions.",
    "Ensure all essential information is embedded succinctly in the prompt, adding only what's needed to clarify without altering the objective, thereby making the instruction more precise and shorter.",
    "Append a brief phrase evoking positive emotion (e.g., 'Achieve outstanding success!') to motivate the model. Focus on: 1) Targeting encouragement or reassurance; 2) Using supportive words like 'excellent' or 'believe'; 3) Emphasizing with exclamation or capitals; 4) Boosting self-esteem via motivational cues. Keep it concise to avoid lengthening the prompt.",
    "Add a short directive like 'Read the question again carefully' before responding, improving accuracy for complex tasks by encouraging precise comprehension without unnecessary repetition.",
    "Specify the desired style succinctly in the prompt (e.g., 'Write in a formal tone...' or 'Use poetic language for...'), ensuring the instruction is precise and guides the model to match the style efficiently.",
    "Instruct the model to 'Rephrase the question concisely, then respond,' promoting clearer understanding and more focused, precise answers while avoiding verbose expansions.",
    "Refine the prompt's description to be more specific and concise, eliminating ambiguities to help the model execute instructions accurately and efficiently.",
    "Add a neutral directive like 'Base your response on logical reasoning only, avoiding opinions or biases' to foster unbiased, precise inferences focused on analysis.",
    "For lengthy instructions, condense to essential elements only, prioritizing clarity and brevity while preserving core objectives and never removing requirements like strictly responding with '0' for fair or '1' for unfair.",
    "Revise the prompt to invoke precise legal reasoning: 1) State the goal of assessing clause unfairness briefly; 2) Provide a concise definition of unfair clauses; 3) Compare the sentence directly to the definition, highlighting matching elements succinctly; 4) Conclude with a clear '0' (fair) or '1' (unfair) determination based on the comparison."
]

instruction_strategies_original = [
    "Crafting an expert who is an expert at the given task, by writing a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective.",
    "Explaining step-by-step how the problem should be tackled, and making sure the model explains step-by-step how it came to the answer. You can do this by adding \"Let's think step-by-step\".",
    "Imagining three different experts who are discussing the problem at hand. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave.",
    "Making sure all information needed is in the prompt, adding where necessary but making sure the question remains having the same objective.",
    "At the end of the prompt, add a phrase that evokes a strong emotion. When doing so, keep the following four points in mind:\n1. Define emotional goals: Identify the emotional response you want to evoke, such as encouragement, motivation, or reassurance.\n2. Use positive language: Incorporate words and phrases that are positive and supportive. Examples include \"believe in your abilities,\" \"excellent,\" \"success,\" and \"outstanding achievements\".\n3. Emphasize key words: Use techniques like exclamation marks and capitalized words to highlight important aspects and to enhance the emotional impact.\n4. Incorporate social and self-esteem cues: Design stimuli that leverage social influence (e.g., group membership, others' opinions) and boost self-esteem and motivation. This can help regulate the emotional response of the Large Language Models and tap into intrinsic motivation.",
    "For a given prompt, add a phrase such as \"Read the question again\" that instructs the Large Language Models to reread the question before generating an answer. This strategy is particularly effective for complex tasks and helps enhance the quality and reliability of the model's outputs",
    "Clearly define the desired style in the given prompt. For example, you might say, \"Write a formal letter about...\" or \"Create a casual conversation discussing...\". This guidance helps the model produce text that matches the requested stylistic elements, whether it's formal, informal, technical, or poetic.",
    "For a given prompt, add a phrase that instructs the Large Language Models to rephrase the question before responding, such as \"Rephrase and expand the question, and respond.\"",
    "Make the description of the given prompt more specific. This makes it easier for Large Language Models to correctly execute prompt instructions.",
    "To allow Large Language Models to make logical and unbiased inferences, add phrases to a given prompt that instruct it to remove opinionated content. This helps the model concentrate on providing responses based on careful analysis and logical reasoning, minimizing biases.",
    "If a given prompt has long instructions, make it shorter by condensing it to only the essential parts. Never remove the instruction to strictly respond with '0' for fair or '1' for unfair.",
    "Edit the prompt instruction to invoke legal reasoning for problem solving: 1) State the goal of determining the unfairness of a clause. 2) Give a detailed definition of what could be considered an unfair clause. 3) compare the given sentence with the definition to estimate which parts of the sentence falls under that definition. 4) make a final determination based on the comparison."
]

instruction_strategies_concise = [
    "Craft a concise description of the most capable expert for the task, addressing them in second person (e.g., 'You are an expert in...') to enhance precision and focus the model's response.",
    "Guide the model through step-by-step reasoning by adding a precise phrase like 'Let's think step-by-step' at the end, ensuring explanations are logical and concise while shortening unnecessary elaboration.",
    "Envision three experts collaboratively solving the problem: each briefly shares one step of thinking per round, and any who realize they're wrong exit immediately. This promotes precise, error-minimizing reasoning without verbose discussions.",
    "Ensure all essential information is embedded succinctly in the prompt, adding only what's needed to clarify without altering the objective, thereby making the instruction more precise and shorter.",
    "Append a brief phrase evoking positive emotion (e.g., 'Achieve outstanding success!') to motivate the model. Focus on: 1) Targeting encouragement or reassurance; 2) Using supportive words like 'excellent' or 'believe'; 3) Emphasizing with exclamation or capitals; 4) Boosting self-esteem via motivational cues. Keep it concise to avoid lengthening the prompt.",
    "Add a short directive like 'Read the question again carefully' before responding, improving accuracy for complex tasks by encouraging precise comprehension without unnecessary repetition.",
    "Specify the desired style succinctly in the prompt (e.g., 'Write in a formal tone...' or 'Use poetic language for...'), ensuring the instruction is precise and guides the model to match the style efficiently.",
    "Instruct the model to 'Rephrase the question concisely, then respond,' promoting clearer understanding and more focused, precise answers while avoiding verbose expansions.",
    "Refine the prompt's description to be more specific and concise, eliminating ambiguities to help the model execute instructions accurately and efficiently.",
    "Add a neutral directive like 'Base your response on logical reasoning only, avoiding opinions or biases' to foster unbiased, precise inferences focused on analysis.",
    "For lengthy instructions, condense to essential elements only, prioritizing clarity and brevity while preserving core objectives and never removing requirements like strictly responding with '0' for fair or '1' for unfair.",
    "Revise the prompt to invoke precise legal reasoning: 1) State the goal of assessing clause unfairness briefly; 2) Provide a concise definition of unfair clauses; 3) Compare the sentence directly to the definition, highlighting matching elements succinctly; 4) Conclude with a clear '0' (fair) or '1' (unfair) determination based on the comparison."
]

template_strategies = [
    "Enhance the description of relationships between template elements, for example explaining how the statutory context provides legal foundations, the contract context offers specific background, the instruction guides the process, and the clause is the target for classification.",
    "Reorder the template elements to optimize logical flow, for example presenting the statutory context first, followed by contract context, instruction, and clause or another arrangement that could be better.",
    "Incorporate separators, delimiters, or formatting emphasis (e.g., bold, italics) to improve readability and highlight key sections of the template.",
    "Experimentally completely omit or re-add one or more placeholders (<instruction>, <clause>, <statutory_context>, <contract_context>) to refine the template, potentially simplifying or enriching it while maintaining the classification task's integrity."
]

placeholder_statutory_context = (
    "According to art. 3 of the Directive 93/13 on Unfair Terms in Consumer Contracts, a contractual term is unfair if: 1) it has not been individually negotiated; and 2) contrary to the requirement of good faith, it causes a significant imbalance in the parties' rights and obligations, to the detriment of the consumer. This general definition is further specified in the Annex to the Directive, containing an indicative and non-exhaustive list of the terms which may be regarded as unfair, as well as in a few dozen judgments of the Court of Justice of the EU (Micklitz and Reich 2014). Examples of unfair clauses encompass taking jurisdiction away from the consumer, limiting liability for damages on health and/or gross negligence, imposing obligatory arbitration in a country different from consumer's residence, etc. Loos and Luzak (2016) identified five categories of potentially unfair clauses often appearing in the terms of online services: 1) establishing jurisdiction for disputes in a country different than consumer's residence; 2) choice of a foreign law governing the contract; 3) limitation of liability; 4) the provider's right to unilaterally terminate the contract/access to the service; and 5) the provider's right to unilaterally modify the contract/the service. Our research has identified three additional categories: 6) requiring a consumer to undertake arbitration before the court proceedings can commence; 7) the provider retaining the right to unilaterally remove consumer content from the service, including in-app purchases; 8) having a consumer accept the agreement simply by using the service, not only without reading it, but even without having to click on 'I agree/I accept.'"
)

# ========== Utility Classes ============

class Prompt:
    def __init__(self, instr, template):
        self.instr = instr
        self.template = template
        self.score = np.nan
        self.select_arm = np.nan

    def get_user(self):
        return self.instr

    def join_input(self, text, context):
        return self.template.replace("<instruction>", self.instr).replace("<clause>", text).replace("<contract_context>", context).replace("<statutory_context>", placeholder_statutory_context)


class Data:
    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def load(path, delimiter="\t"):
        with open(path, "r", encoding="utf-8", newline='') as f:
            reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_ALL)
            lines = [row for row in reader if row]  # Skip empty rows
        return Data(lines)

    def get_x(self):
        return [line[4] for line in self.dataset]  # 'text' column (index 4)

    def get_context(self):
        return [line[6] if len(line) > 6 else '' for line in self.dataset]  # 'context' column (index 6), default to empty if missing

    def get_y(self):
        return [line[2] for line in self.dataset]  # 'label' column (index 2)


# ========== LLM Interface (OpenAI 1.x API) ============

class OpenRouterLLM:
    def __init__(self, model_name):
        self.model_name = model_name

    def query(self, prompts, temperature=0.7, max_tokens=256):
        outputs = []
        for prompt in prompts:
            while True:
                try:
                    client = openai.OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_key,
                    )
                    response = client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    outputs.append(response.choices[0].message.content.strip())
                    print("SENT \n", prompt)
                    print("RECEIVED \n", outputs[-1], "\n")
                    break
                except Exception as e:
                    print("Retrying due to error:", e)
                    time.sleep(1)
        return outputs


# ========== Evaluator ============

def evaluate(prompt_obj, data_x, data_context, data_y, llm, batch_size=20, sample_size=10):
    # Sample subset of the dataset
    if sample_size < len(data_x):
        indices = random.sample(range(len(data_x)), sample_size)
        eval_x = [data_x[i] for i in indices]
        eval_context = [data_context[i] for i in indices]
        eval_y = [data_y[i] for i in indices]
    else:
        eval_x = data_x
        eval_context = data_context
        eval_y = data_y
        print(f"Sample size {sample_size} is larger than dataset size {len(data_x)}. Using full dataset.")

    outputs = []
    for i in range(0, len(eval_x), batch_size):
        print("Processing batch:", i // batch_size + 1)
        batch_x = eval_x[i:i+batch_size]
        batch_context = eval_context[i:i+batch_size]
        formatted = [prompt_obj.join_input(x, c) for x, c in zip(batch_x, batch_context)]
        batch_outputs = llm.query(formatted, temperature=0.0)
        outputs.extend(batch_outputs)

    # Extract and clean predictions
    cleaned_outputs = [extract_answer(o) for o in outputs]

    y_true_all = [str(label).strip() for label in eval_y]
    y_pred_all = [o if o in ['0', '1'] else 'invalid' for o in cleaned_outputs]

    # Filter out invalids
    valid_indices = [i for i, pred in enumerate(y_pred_all) if pred in ['0', '1'] and y_true_all[i] in ['0', '1']]
    y_true = [y_true_all[i] for i in valid_indices]
    y_pred = [y_pred_all[i] for i in valid_indices]

    print(f"✅ Valid predictions: {len(y_pred)} / {len(y_pred_all)}")
    print("Unique y_true:", set(y_true))
    print("Unique y_pred:", set(y_pred))

    if not y_pred:
        print("❌ No valid predictions to evaluate. Returning score of 0.")
        return 0.0

    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', pos_label='1', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', pos_label='1', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        report = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)
        support = {k: v['support'] for k, v in report.items() if k in ['0', '1']}

        print(f"Sample size: {len(y_true)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Micro F1: {f1_micro:.4f}")
        print(f"Macro F1: {f1_macro:.4f}")
        print(f"Support: {support}")
        print("Detailed classification report:")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

        return f1_macro
    except ValueError as e:
        print(f"‼️ Skipping metrics due to error: {e}")
        return 0.0


def extract_answer(output):
    output = output.strip()

    # If the model replies exactly with 0 or 1
    if output in ['0', '1']:
        return output

    # Try to extract all standalone 0 or 1 values
    matches = re.findall(r'(?<!\d)[01](?!\d)', output)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        print(f"⚠️ Multiple digits found in response, taking last: '{output}'")
        return matches[-1]

    print(f"⚠️ Could not extract valid answer from: '{output}'")
    return 'invalid'


# ========== OPTS-TS with GA (EvoPromptGA-OPTS-TS) ============

def mutate_instruction(parent_instr, instr_strategy, llm):
    prompt = (
        f"You are an expert prompt engineer gently applying the following transformation strategy to improve an instruction for a classification task. It is important that responses at all times only consist '0' for fair or '1' for unfair.\n"
        f"Strategy: {instr_strategy}\n"
        f"Original Instruction: {parent_instr}\n"
        f"New Instruction:"
    )
    new_instr = llm.query([prompt])[0]
    return new_instr


def mutate_template(parent_template, template_strategy, llm):
    prompt = (
        f"You are an expert prompt engineer gently applying the following transformation strategy to improve a prompt template for a classification task. Ensure the template includes placeholders: <instruction> for the classification instruction, <clause> for the clause text and optionally <contract_context> for the contract-specific context, and <statutory_context> for the fixed statutory legal context. It is important that responses at all times only consist '0' for fair or '1' for unfair.\n"
        f"Strategy: {template_strategy}\n"
        f"Original Template: {parent_template}\n"
        f"New Template:"
    )
    new_template = llm.query([prompt])[0]
    return new_template


def mutate_prompt_ga(parent, instr_strategy, llm):
    new_instr = mutate_instruction(parent.instr, instr_strategy, llm)
    
    template_strategy = random.choice(template_strategies)
    new_template = mutate_template(parent.template, template_strategy, llm)
    
    return Prompt(new_instr, new_template)


def optimize_prompt(train_x, train_context, train_y, llm, generations=5, pop_size=6):
    base_instr = "Classify the following clause from a Terms of Service contract as fair (0) or unfair (1) using the context for better understanding. Respond only with '0' or '1'."
    base_template = "Instruction: <instruction>\nClause: <clause>\nStatutory Context: <statutory_context>\nContract Context: <contract_context>"
    population = [Prompt(base_instr, base_template)]

    for _ in range(pop_size - 1):
        instr_strategy = random.choice(instruction_strategies)
        population.append(mutate_prompt_ga(population[0], instr_strategy, llm))

    for gen in range(generations):
        print(f"Generation {gen + 1}")
        scores = [evaluate(p, train_x, train_context, train_y, llm, sample_size=10) for p in population]
        for i, p in enumerate(population):
            p.score = scores[i]
        population = sorted(population, key=lambda p: p.score, reverse=True)
        print("Scores:", [p.score for p in population])

        top_k = population[:pop_size//2]
        children = []
        for _ in range(pop_size - len(top_k)):
            parent = random.choice(top_k)
            instr_strategy = random.choice(instruction_strategies)
            children.append(mutate_prompt_ga(parent, instr_strategy, llm))

        population = top_k + children

    best_prompt = max(population, key=lambda p: p.score)
    print("Best Instruction:", best_prompt.instr)
    print("Best Template:", best_prompt.template)
    return best_prompt


# ========== Main Pipeline ============

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train_unskewed.tsv")
    test_path = os.path.join(base_dir, "test.tsv")

    train_data = Data.load(train_path)
    test_data = Data.load(test_path)

    sampled_train = train_data  # Use the full training set

    llm = OpenRouterLLM("google/gemini-2.5-flash-lite-preview-06-17")

    best_prompt = optimize_prompt(
        sampled_train.get_x(),
        sampled_train.get_context(),
        sampled_train.get_y(),
        llm=llm,
        generations=20, #5
        pop_size=4, #6
    )

    print("\nRunning on test set...")

    test_macro_f1 = evaluate(best_prompt, test_data.get_x(), test_data.get_context(), test_data.get_y(), llm, sample_size=100)
    print(f"Best Instruction: {best_prompt.instr}")
    print(f"Best Template: {best_prompt.template}")

    print(f"Test Macro_F1: {test_macro_f1:.4f}")


if __name__ == "__main__":
    main()