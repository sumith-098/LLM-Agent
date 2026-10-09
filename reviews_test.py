import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"   # agar error aaye toh dashboard wale Embedding 2 ka ID try karenge

REVIEWS = [
    "Interstellar ka ending bahut confusing tha, kuch samajh nahi aaya",
    "Last 20 minute mein main bilkul lost ho gaya, par visuals kamaal the",
    "Interstellar ne mujhe rula diya, father-daughter wala scene dil chhu gaya",
    "Hera Pheri ne pet dukha diya hasi se, full paisa vasool comedy",
    "Dune ki cinematography zabardast hai par pace bahut slow hai",
    "Arrival chhoti si movie hai par dimaag ghuma deti hai, emotional bhi",
]

def embed(texts, task):
    res = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,                      # poori list ek hi call mein (batch)
        config=types.EmbedContentConfig(task_type=task),
    )
    return np.array([e.values for e in res.embeddings])

doc_vecs = embed(REVIEWS, "RETRIEVAL_DOCUMENT")      # 1 request mein saare reviews

def semantic_search(query, k=6):
    q = embed([query], "RETRIEVAL_QUERY")[0]
    # cosine similarity 
    sims = doc_vecs @ q / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(q))
    top = sims.argsort()[::-1][:k]
    return [(round(float(sims[i]), 5), REVIEWS[i]) for i in top]

def keyword_search(query):
    words = query.lower().split()
    return [r for r in REVIEWS if any(w in r.lower() for w in words)]

query = "confusing story"
print("KEYWORD :", keyword_search(query))
print("SEMANTIC:")
for score, text in semantic_search(query):
    print(f"  {score}  {text}")