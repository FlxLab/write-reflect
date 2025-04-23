"""
Modular Q&A Companion (Terminal UI)

This script allows the user to ask reflective questions to their writing archive.
It uses semantic retrieval + a prompt-formatted LLM call to return grounded reflections.

Dependencies:
- embedded_chunks.pkl (with id, text, tags, reasoning, embedding)
- utils.py (shared functions)
- Ollama running (e.g. LLaMA3, Mistral, etc.)
"""

from utils import load_archive, embed_query, get_top_chunks, format_chunks_as_context, format_chunks_for_qa, query_llm




# Load archive
data = load_archive("embedded_chunks.pkl")

print("\nWelcome to the Modular Q&A Companion 💬")
print("Ask reflective questions to explore your archive.\n")

while True:
    user_query = input("Your question (or type 'done' to exit):\n> ")
    if user_query.strip().lower() in ["done", "n", "exit"]:
        break

    print("\nGenerating response...\n")

    # Embed, retrieve, build context
    query_embedding = embed_query(user_query)
    top_chunks = get_top_chunks(query_embedding, data)
    prompt = format_chunks_as_context(top_chunks, user_query)

    # Call LLM
    response = query_llm(prompt)

    print("\n--- Reflective Response ---\n")
    print(response)
    print("\n--- End ---\n")
