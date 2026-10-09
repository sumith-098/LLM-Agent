import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-3.1-flash-lite"

# ---------- Mock CineBook data (sample) ----------
MOVIES = [
    {"title": "Interstellar", "genre": "sci-fi", "minutes": 169},
    {"title": "Inception",    "genre": "sci-fi", "minutes": 148},
    {"title": "The Martian",  "genre": "sci-fi", "minutes": 144},
    {"title": "Arrival",      "genre": "sci-fi", "minutes": 116},
    {"title": "Dune",         "genre": "sci-fi", "minutes": 155},
    {"title": "Hera Pheri",   "genre": "comedy", "minutes": 156},
]

# ---------- TOOLS ----------
def search_movies(genre: str, max_minutes: int) -> str:
    """Search the CineBook catalog by genre (e.g. 'sci-fi', 'comedy')
    and maximum duration in minutes. Returns matching movies with durations."""
    hits = [m for m in MOVIES
            if m["genre"] == genre.lower() and m["minutes"] <= max_minutes]
    return str(hits) if hits else "No movies found"

def calculator(expression: str) -> str:
    """Evaluate a math expression like '148 * 3'."""
    if not set(expression) <= set("0123456789+-*/(). "):
        return "Invalid expression"
    return str(eval(expression))

TOOLS = {"search_movies": search_movies, "calculator": calculator}

config = types.GenerateContentConfig(
    tools=list(TOOLS.values()),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# ---------- AGENT LOOP --------
def run_agent(question: str, max_steps: int = 5):       
    print(f"\n🧑 USER: {question}\n")
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    for step in range(1, max_steps + 1):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )
        contents.append(response.candidates[0].content)   # model ka turn history mein

        calls = response.function_calls
        if not calls:                                      # tool nahi chahiye = final answer
            print(f"🤖 FINAL ANSWER:\n{response.text}")
            return

        result_parts = []
        for call in calls:
            print(f"🔧 Step {step}: agent ne {call.name}({dict(call.args)}) call kiya")
            result = TOOLS[call.name](**call.args)
            print(f"   ↳ result: {result}\n")
            result_parts.append(
                types.Part.from_function_response(
                    name=call.name, response={"result": result}
                )
            )
        contents.append(types.Content(role="user", parts=result_parts))

    print("⚠️ max_steps khatam, agent ruk gaya")

run_agent(
    "Mujhe 2.5 hours ke andar koi sci-fi movie suggest karo, "
    "aur agar main sabse lambi wali movie 3 baar dekhun toh total kitne minutes lagenge?"
)