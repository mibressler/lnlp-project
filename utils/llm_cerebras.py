# Models and Limits: https://cloud.cerebras.ai/platform/org_9hewjr6yrdh8rjvm5x4fy8et/models 
# Documentation: https://inference-docs.cerebras.ai/api-reference/chat-completions

# For more free api resources, visit: https://github.com/cheahjs/free-llm-api-resources

import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import time

load_dotenv()

# messages = [{"role": "user OR system OR assistant", "content": message}, ...]
def get_llm_response(messages, model="llama-3.3-70b"): # Default model is Llama 3.3 70B => 30 requests per minute
    client = Cerebras(
        api_key=os.environ.get("CEREBRAS_API_KEY"),
    )
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    time.sleep(2) # To avoid hitting rate limits, add a delay after the request
    return chat_completion.choices[0].message

def get_llm_task_response(user_prompt: str, system_prompt: str, few_shot_examples="", model="llama-3.3-70b"):
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
