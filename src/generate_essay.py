"""
generate_essay_modular.py

Modular version of the essay builder. Uses functions from utils.py.
Builds reflective essays one paragraph at a time using your AI & Africa archive.
Each section is generated based on semantically retrieved chunks and a polished prompt.

Includes automatic saving of the final essay + follow-up questions to user-specified .txt files.
"""

import os
from utils import load_archive, embed_query, get_top_chunks, format_chunks_as_context, query_llm, project_path

# Load embedded archive (DataFrame with id, text, tags, reasoning, embedding)
data = load_archive(project_path("data", "processed", "embedded_chunks.pkl"))



# Store the essay parts and follow-ups
essay_sections = []
followups = []

print("\nWelcome to the Modular Essay Builder ✍️")
print("Build a reflective essay one part at a time.\n")

while True:
    query = input("Enter your next essay prompt or section idea (or type 'done' to finish):\n> ")
    if query.strip().lower() in ["done", "n"]:
        break

    print("\nGenerating essay section...\n")
    query_embedding = embed_query(query)
    top_chunks = get_top_chunks(query_embedding, data)
    prompt = format_chunks_as_context(top_chunks, query)
    result = query_llm(prompt)

    # Extract essay section and follow-ups using marker
    if "--- FOLLOW-UP-BEGIN ---" in result:
        essay_body, followup_block = result.split("--- FOLLOW-UP-BEGIN ---", 1)
        section_text = f"[Prompt: {query}]\n{essay_body.strip()}"
        fups = followup_block.strip()
    else:
        section_text = f"[Prompt: {query}]\n{result.strip()}"
        fups = ""

    essay_sections.append(section_text)
    if fups:
        followups.append(fups)

    print("\n--- Essay Section ---\n")
    print(section_text)
    if fups:
        print("\nFollow-up Questions:\n")
        print(fups)

# Save outputs
if essay_sections:
    print("\nEssay complete. Where should I save the output?")
    base_path = input("→ Enter folder path or press Enter to use current folder: ").strip()
    filename = input("→ Enter filename base (no extension): ").strip()

    if not base_path:
        base_path = os.getcwd()
    os.makedirs(base_path, exist_ok=True)

    essay_path = os.path.join(base_path, f"{filename}.txt")
    fup_path = os.path.join(base_path, f"{filename}_followups.txt")

    final_essay = "\n\n".join(essay_sections)

    with open(essay_path, "w", encoding="utf-8") as f:
        f.write(final_essay)

    with open(fup_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(followups))

    print(f"\n✓ Essay saved to: {essay_path}")
    print(f"✓ Follow-up questions saved to: {fup_path}")
else:
    print("\nNo essay content was generated.")
