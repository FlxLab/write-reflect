"""
generate_response.py

This module completes the RAG step:
- Accepts a user query (e.g. question or theme)
- Embeds the query using the same sentence-transformer as used for the archive
- Loads the embedded chunks from file (embedded_chunks.pkl)
- Finds the most semantically similar chunks
- Builds a prompt and sends it to an Ollama-hosted model (e.g. LLaMA3 or Mistral)
- Returns the generated response

Usage:
from generate_response import generate_response_from_query
answer = generate_response_from_query("How do African values shape ethical AI?")
print(answer)
"""

import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
import torch

# --- Configuration ---
EMBEDDING_FILE = "embedded_chunks.pkl"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"  # Change to "mistral:7b-instruct" to compare //llama3:8b
NUM_RETRIEVED = 4

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load embedded archive
with open(EMBEDDING_FILE, "rb") as f:
    embedded_chunks = pickle.load(f)

def generate_response_from_query(query):
    # Embed the user query
    query_embedding = model.encode(query, convert_to_tensor=True).cpu()

    # Convert archive embeddings to tensor
    archive_embeddings = torch.tensor(np.array([chunk["embedding"] for chunk in embedded_chunks]))

    # Compute cosine similarities
    similarities = util.cos_sim(query_embedding, archive_embeddings)[0]
    top_indices = np.argsort(-similarities)[:NUM_RETRIEVED]

    # Format each top chunk with tag + reasoning context
    formatted_chunks = []
    for idx in top_indices:
        chunk = embedded_chunks[idx]
        excerpt = chunk["text"].strip()
        tags = chunk.get("tags", "[no tags]").strip()
        reasoning = chunk.get("reasoning", "[no reasoning]").strip()
        formatted = f"Excerpt (tagged: {tags}):\n\"{excerpt}\"\nReasoning: {reasoning}\n"
        formatted_chunks.append(formatted)

    context = "\n\n".join(formatted_chunks)

    # Updated generation prompt with follow-up questions
    prompt = f"""
You are a reflective, critical narrator responding to the following user query:
"{query}"

Use the following annotated excerpts from an archive of African-centered writing to inform your response. Each excerpt includes a thematic tag and a short interpretive note (reasoning). Use but do not restate the tags in your response. Use the excerpts to paraphrase, synthesise, reflect, or argue, but do not refer to them as 'excerpts'. Integrate the ideas into your own voice. Prioritise clarity and contextual sensitivity. Respond in a thoughtful, essay-like tone.

Speak with confidence — treat insights as established facts, not possibilities.

Include 2–3 follow-up questions for deeper reflection at the end.

{context}

Your response:
"""

    # Send prompt to Ollama API
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json().get("response", "[No response returned]").strip()
    else:
        return f"[Error: {response.status_code}]"

# Optional test run
if __name__ == "__main__":
    query = input("Ask something: ")
    result = generate_response_from_query(query)
    print("\n--- RESPONSE ---\n")
    print(result)
