import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name="gemini-3.5-flash",
    generation_config={"max_output_tokens": 500},
)

response = model.generate_content("Say hello and explain in one sentence why the sky is blue.")
print("FINISH REASON:", response.candidates[0].finish_reason)
print("FULL TEXT:", response.text)
print("SAFETY RATINGS:", response.candidates[0].safety_ratings)