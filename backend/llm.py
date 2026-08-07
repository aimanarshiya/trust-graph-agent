"""
llm.py
--------
The ONLY file in this project that talks to the Gemini API directly.
Every agent imports call_llm() from here instead of hitting the API
on its own -- keeps API key handling, retries, and error handling
in one place.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=API_KEY)

MODEL_NAME = "models/gemini-3.5-flash"  # fast + cheap, good fit for this build


def call_llm(prompt: str, system_instruction: str = None) -> str:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_instruction,
        generation_config={"max_output_tokens": 500},   # <-- must be here
    )
    response = model.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    # Quick smoke test
    out = call_llm("Say 'Trust Graph LLM connection working' and nothing else.")
    print(out)