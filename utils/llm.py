# Models and pricing: https://openrouter.ai
# API reference: https://openrouter.ai/docs

"""Lightweight helper for making LLM calls via OpenRouter."""

# For additional free API resources see:
# https://github.com/cheahjs/free-llm-api-resources

import os
import time
from typing import Dict, List

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Automatically load environment variables from a `.env` file located anywhere
# up the directory tree.  This allows running scripts from subfolders while
# still picking up the project-level configuration.
load_dotenv(find_dotenv())

DEFAULT_MODEL = "mistralai/mistral-7b-instruct"

# messages = [{"role": "user" | "system" | "assistant", "content": message}, ...]
def get_llm_response(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL):
    """Send a chat completion request to OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )

    time.sleep(0.1)  # Avoid hitting rate limits
    return chat_completion.choices[0].message

def get_llm_task_response(
    user_prompt: str,
    system_prompt: str,
    few_shot_examples: str = "",
    model: str = DEFAULT_MODEL,
):
    """
    Get a response from the LLM for a specific task.
    This function is a wrapper around get_llm_response to handle task-specific messages.
    """
    messages = [
        {"role": "system", "content": system_prompt + '\n'},
        {"role": "user", "content": few_shot_examples + '\n' + user_prompt}
    ]

    return get_llm_response(messages, model=model)

# Example usage:
if __name__ == "__main__":
    # pass
    messages = [
        {"role": "system", "content": "Answer with single word: FAIR or UNFAIR. Read the sentence and judge if it is fair or not."},
        {"role": "user", "content": "A person is judged to be guilty of a crime without a fair trial."},
    ]
    response = get_llm_response(messages)
    print("Response:", response.content)
