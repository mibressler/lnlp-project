import json
import re
from typing import Optional

def extract_json_from_text(response: str) -> Optional[str]: # if not found return None
    try:
        # Look for the first JSON object in the text
        json_match = re.search(r'\{[\s\S]*?\}', response)
        if not json_match:
            return None

        # Try parsing the matched JSON
        data = json.loads(json_match.group())
        return data
    except Exception:
        return None