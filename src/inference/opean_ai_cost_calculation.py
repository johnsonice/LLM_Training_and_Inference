#!/usr/bin/env python3
"""
Simple OpenAI completion with real-time cost calculation.
Self-contained module with no external dependencies.
"""
#%%
import os
from openai import OpenAI
from dotenv import load_dotenv
from openai_cost_calculator import estimate_cost
import sys
import pathlib

# Add the 'libs' folder to sys.path for local imports
libs_path = pathlib.Path(__file__).parent.parent / "libs"
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

load_dotenv()
#%%
def extract_token_breakdown(response):
    """
    Given an OpenAI Response object, returns a dict with:
      - prompt_tokens: total prompt tokens (including cached)
      - cached_tokens: number of cached tokens (if any)
      - completion_tokens: output tokens generated
    """
    usage = response.usage  # ResponseUsage object

    prompt_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None))
    completion_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None))

    # Extract nested details for cached tokens
    cached_tokens = None
    if hasattr(usage, "input_tokens_details"):
        cached_tokens = getattr(usage.input_tokens_details, "cached_tokens", 0)
    elif hasattr(usage, "prompt_tokens_details"):
        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0)

    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens
    }
#%%
if __name__ == "__main__":
    # Example usage
    prompt = "Pleage generate 100 words"
    model_name = "o3"
    reasoning_effort = {"effort": "medium"} #"medium"
    #%%\
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    response = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
        reasoning=reasoning_effort,
        temperature=None
    )
    #%%
    print(model_name)
    print(estimate_cost(response))
    print(extract_token_breakdown(response))
    print(response.output_text)

# %%
