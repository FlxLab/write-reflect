"""
embed_chunks.py

This script uses the 'sentence-transformers' library to convert each chunk of text
(from the labeled file 'writing_chunks_labeled.csv') into a semantic vector (embedding).

The resulting list of embeddings and their corresponding metadata (id, text, tags, reasoning)
are saved into a pickle file 'embedded_chunks.pkl'.

Model used:
- all-MiniLM-L6-v2 (a general-purpose transformer for sentence similarity)

This enables future semantic search: finding the most relevant text chunks for a given user query.
"""

from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle

# Load the labeled chunks
input_path = "writing_chunks_labeled.csv"
df = pd.read_csv(input_path)

# Load the sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract just the text chunks
texts = df["text"].tolist()

# Generate embeddings for each chunk
embeddings = model.encode(texts, show_progress_bar=True)

# Combine embeddings with other data
embedded_data = []
for i, row in df.iterrows():
    embedded_data.append({
        "id": row["id"],
        "text": row["text"],
        "tags": row["tags"],
        "reasoning": row["reasoning"],
        "embedding": embeddings[i]
    })

# Save to pickle for later use
output_path = "embedded_chunks.pkl"
with open(output_path, "wb") as f:
    pickle.dump(embedded_data, f)

print(f"\n Embedded {len(embedded_data)} chunks and saved to: {output_path}")
