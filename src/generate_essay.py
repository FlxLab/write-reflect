"""
generate_essay.py

Interactive essay builder that uses your AI + Africa text archive to generate paragraph-by-paragraph reflective writing.
Each paragraph is generated using semantically retrieved excerpts based on your input prompt.
The script now includes automatic saving of the final essay and follow-up questions to user-specified .txt files.

Dependencies:
- embedded_chunks.pkl (must contain id, text, tags, reasoning, and embedding)
- Ollama running with LLaMA3 or other local model
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import requests
import time

followups = []  # global list, will store all follow-up questions across essay sections

# Load data
raw_data = pd.read_pickle("embedded_chunks.pkl")
data = pd.DataFrame(raw_data)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ollama setup
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

def generate_essay_section(query, num_chunks=5):
    query_embedding = model.encode(query, convert_to_tensor=True).cpu()

    archive_embeddings = torch.stack([
        torch.tensor(entry, dtype=torch.float32) for entry in data["embedding"]
    ]).cpu()

    similarities = util.cos_sim(query_embedding, archive_embeddings)[0]
    top_indices = similarities.topk(num_chunks).indices.tolist()
    retrieved = data.iloc[top_indices]

    context = "\n\n".join([
        f"{entry['text']}\n(Tag: {entry['tags']})\n{entry['reasoning']}"
        for _, entry in retrieved.iterrows()
    ])

    prompt = f"""
You are a reflective, critical narrator responding to the following user query:
"{query}"

Use the following annotated excerpts from an archive of African-centered writing to inform your response. Each excerpt includes a thematic tag and a short interpretive note (reasoning). Use but do not restate the tags in your response. Use the excerpts to paraphrase, synthesise, reflect, or argue, but do not refer to them as 'excerpts'. Integrate the ideas into your own voice. Prioritise clarity and contextual sensitivity. Respond in a thoughtful, essay-like tone.

Speak with confidence — treat insights as established facts, not possibilities.

Include 2–3 follow-up questions for deeper reflection at the end.

{context}

Your response:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        response_text = response.json().get("response", "[No response returned]").strip()
        # Extract follow-up questions (lines that start with 1., 2., etc.)
        extracted = []
        for line in response_text.splitlines():
            if line.strip().startswith(("1.", "2.", "3.")):
                extracted.append(line.strip())
        # Add to global followups list
        if extracted:
            followups.extend(extracted)

            return response_text
        else:
            return f"[Error: HTTP {response.status_code}]"
        
if __name__ == "__main__":
    print("\nWelcome to the Essay Builder ✍️")
    print("Build a reflective essay one part at a time.")

    section_texts = []
    follow_up_questions = []

    while True:
        query = input("\nEnter your next essay question or section idea (or type 'done' to finish):\n> ")
        if query.strip().lower() in ["done", "n"]:
            break

        print("\nGenerating essay section... please wait...\n")
        section = generate_essay_section(query)
        print("\n--- Essay Section ---\n")
        print(section)

        section_texts.append(f"[Prompt: {query}]\n{section}")

        # Extract follow-up questions (after last line break)
        lines = section.strip().splitlines()
        fups = [line for line in lines if line.strip().startswith("1.")]
        if fups:
            follow_up_questions.extend(lines[-3:])

# After user types 'done', save the essay and follow-up questions
print("\nEssay complete.")

# Ask for full save path or leave blank
save_path = input("→ Optional: Enter full folder path to save files (or press Enter to save in current directory): ").strip()

# Ask for filename (no extension)
filename_base = input("→ Enter filename base (no extension): ").strip()
if not filename_base:
    filename_base = "essay_output"

# Create full file paths
if save_path:
    essay_path = f"{save_path.rstrip('/')}/{filename_base}.txt"
    fup_path = f"{save_path.rstrip('/')}/{filename_base}_followups.txt"
else:
    essay_path = f"{filename_base}.txt"
    fup_path = f"{filename_base}_followups.txt"

 # Combine all essay parts into final string
final_essay = "\n\n".join(section_texts)
   

# Save full essay
with open(essay_path, "w", encoding="utf-8") as f:
    f.write(final_essay)

# Save follow-up questions
with open(fup_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(followups))

print(f"\n✅ Essay saved to: {essay_path}")
print(f"✅ Follow-up questions saved to: {fup_path}")

