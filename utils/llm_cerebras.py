# Models and Limits: https://cloud.cerebras.ai/platform/org_9hewjr6yrdh8rjvm5x4fy8et/models 

import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

# messages = [{"role": "user OR system OR assistant", "content": message}, ...]
def get_llm_response(messages, model="llama-3.3-70b"):
    client = Cerebras(
        api_key=os.environ.get("CEREBRAS_API_KEY"),
    )
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message

# Example usage:
if __name__ == "__main__":
    pass
    # response = get_llm_response("Why is fast inference important?")
    # print("Response:", response)
