"""
utils.py

This module contains reusable utility functions for the Critical AI Writing Companion.
These can be used by multiple scripts such as generate_response_modular.py, generate_essay.py,
and Gradio interfaces.
"""

import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import requests

# Load the embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3:8b"

def load_archive(pickle_path):
    """Load archived data with embeddings, tags, and reasoning."""
    # IMPORTANT: pickle_path must be passed relative to the script location (e.g. "../data/processed/embedded_chunks.pkl")
    raw = pd.read_pickle(pickle_path)
    df = pd.DataFrame(raw)
    return df

def embed_query(query):
    """Convert a query string to a sentence embedding tensor."""
    return embedding_model.encode(query, convert_to_tensor=True).cpu()

def get_top_chunks(query_embedding, df, num_chunks=5):
    """Retrieve top-N semantically similar archive chunks based on cosine similarity."""
    archive_embeddings = torch.stack([
        torch.tensor(e, dtype=torch.float32) for e in df["embedding"]
    ]).cpu()

    similarities = util.cos_sim(query_embedding, archive_embeddings)[0]
    top_indices = similarities.topk(num_chunks).indices.tolist()
    return df.iloc[top_indices].to_dict("records")


def format_chunks_as_context(chunks, query):
    """
    Construct a complete prompt to send to the LLM.
    Includes instructions, the user query, and selected archive chunks.

    Adds a unique marker (--- FOLLOW-UP-BEGIN ---) before the model is asked to generate questions.
    This allows accurate and consistent extraction of follow-ups.
    """
    formatted_chunks = "\n\n".join([
        f"{chunk['text']}\n(Tag: {chunk['tags']})\n{chunk['reasoning'].strip()}"
        for chunk in chunks
    ])

    prompt = f"""
You are a reflective, critical narrator responding to the following user query:
"{query}"

Use the following annotated excerpts from an archive of African-centered writing to inform your response. Each excerpt includes a thematic tag and a short interpretive note (reasoning). Use but do not restate the tags in your response. Use the excerpts to paraphrase, synthesise, reflect, or argue, but do not refer to them as 'excerpts'. Integrate the ideas into your own voice. Prioritise clarity and contextual sensitivity. Respond in a thoughtful, essay-like tone.

Speak with confidence — treat insights as established facts, not possibilities.

After your response, include the following marker exactly as written:
--- FOLLOW-UP-BEGIN ---

Then provide 2–3 follow-up questions for deeper reflection, one per line and numbered.

{formatted_chunks}

Your response:
"""
    return prompt.strip()

def format_chunks_for_qa(chunks, query):
    """
    Prompt for conversational Q&A. Responds reflectively to a user’s question.
    Draws insight from provided archive texts, without revealing that fact.
    """

    formatted_chunks = "\n\n".join([
        f"{chunk['text']}\n(Tag: {chunk['tags']})\n{chunk['reasoning'].strip()}"
        for chunk in chunks
    ])

    prompt = f"""
You are a reflective, critical narrator responding to the following question:
"{query}"

Offer a thoughtful, grounded response based on the ideas below. Speak in your own voice — do not mention where these ideas came from. Integrate key themes, insights, and connections, but do not reference any 'excerpts', 'texts', or sources.

Respond in a warm, intelligent tone. Use 1–2 short paragraphs.

At the end, write exactly:
--- FOLLOW-UP-BEGIN ---

Then provide 2–3 follow-up questions for deeper thought.

{formatted_chunks}

Your response:
"""
    return prompt.strip()



def query_llm(prompt, model=DEFAULT_MODEL):
    """Send a prompt to the LLM running via Ollama and return the response."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    if response.status_code == 200:
        return response.json().get("response", "[No response returned]").strip()
    else:
        return f"[Error: HTTP {response.status_code} – check if Ollama is running?]"
