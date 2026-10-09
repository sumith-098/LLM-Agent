import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def calculator(expression: str) -> str:
    """Evaluate a simple math expression like '25 * 4 + 10'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Invalid expression"
    return str(eval(expression))

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="What is 1234 * 5678 + 99?",
    config=types.GenerateContentConfig(tools=[calculator]),  #types.GenerateContentConfig(
                                                                            # temperature=0.2,               kitna creative/random
                                                                            # max_output_tokens=500,          answer kitna lamba
                                                                            # system_instruction="You are a helpful math tutor.",
                                                                         # tools=[calculator],              kaunse tools available hain
                                                                                           #)
)
print(response.text)