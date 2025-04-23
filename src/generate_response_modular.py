"""
generate_response_modular.py

Terminal-based Q&A Companion using your AI + Africa archive.

This script allows you to type a reflective question and receive a short, thoughtful response drawn from your archive.
It uses the utils.py module for all core logic (embedding, similarity search, formatting, LLM call).
"""

from utils import load_archive, embed_query, get_top_chunks, format_chunks_for_qa, query_llm

# Load archive
data = load_archive("embedded_chunks.pkl")

print("\nWelcome to the Modular Q&A Companion 💬")
print("Ask reflective questions to explore your archive.\n")

while True:
    query = input("Your question (or type 'done' to exit):\n> ")
    if query.strip().lower() in ["done", "n", "exit", "quit"]:
        break

    print("\nGenerating response...\n")
    query_embedding = embed_query(query)
    top_chunks = get_top_chunks(query_embedding, data)
    prompt = format_chunks_for_qa(top_chunks, query)
    result = query_llm(prompt)

    # Split response into main part and follow-up questions
    if "--- FOLLOW-UP-BEGIN ---" in result:
        main_text, followups = result.split("--- FOLLOW-UP-BEGIN ---", 1)
        print("\n--- Reflective Response ---\n")
        print(main_text.strip())
        print("\n--- Follow-up Questions ---\n")
        print(followups.strip())
    else:
        print("\n--- Reflective Response ---\n")
        print(result.strip())
