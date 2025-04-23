"""
gradio_interface.py

Interactive Gradio UI for the Critical AI Writing Companion.
Supports:
- Essay Builder: build reflective essays section-by-section.
- Q&A Companion: ask one-off reflective questions.
"""

import gradio as gr
import os
from utils import (
    load_archive,
    embed_query,
    get_top_chunks,
    format_chunks_as_context,
    format_chunks_for_qa,
    query_llm
)

# Load data
data = load_archive("embedded_chunks.pkl")

# Session storage for Essay Builder
essay_sections = []
followup_list = []

def handle_generate(mode, user_input, current_essay, current_fups):
    if not user_input.strip():
        return current_essay, current_fups, "Please enter a prompt."

    query_embedding = embed_query(user_input)
    top_chunks = get_top_chunks(query_embedding, data)

    if mode == "Essay Builder":
        prompt = format_chunks_as_context(top_chunks, user_input)
    else:
        prompt = format_chunks_for_qa(top_chunks, user_input)

    result = query_llm(prompt)

    if "--- FOLLOW-UP-BEGIN ---" in result:
        main, fups = result.split("--- FOLLOW-UP-BEGIN ---", 1)
        main, fups = main.strip(), fups.strip()
    else:
        main, fups = result.strip(), ""

    if mode == "Essay Builder":
        section = f"[Prompt: {user_input}]\n{main}"
        essay_sections.append(section)
        if fups:
            followup_list.append(fups)

        updated_essay = "\n\n".join(essay_sections)
        updated_fups = "\n\n".join(followup_list)
        return updated_essay, updated_fups, "✓ Section added."
    else:
        return main, fups, "✓ Answer generated."

def handle_finish(filename):
    if not essay_sections:
        return "No essay to save yet."

    base_path = os.getcwd()
    essay_path = os.path.join(base_path, f"{filename}.txt")
    fup_path = os.path.join(base_path, f"{filename}_followups.txt")

    with open(essay_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(essay_sections))

    with open(fup_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(followup_list))

    return f"✓ Essay saved to:\n{essay_path}\n✓ Follow-ups saved to:\n{fup_path}"

def handle_clear():
    return "", "", ""

with gr.Blocks() as demo:
    gr.Markdown("## ✍️ Critical AI Writing Companion (Gradio UI)")

    with gr.Row():
        mode_selector = gr.Radio(["Essay Builder", "Q&A Companion"], value="Essay Builder", label="Choose Mode")

    user_input = gr.Textbox(label="Your question or prompt", lines=3)
    output = gr.Textbox(label="Generated Response", lines=15)
    followups = gr.Textbox(label="Follow-up Questions", lines=6)

    with gr.Row():
        clear_btn = gr.Button("Clear")
        gen_btn = gr.Button("Generate")
        finish_btn = gr.Button("Finish Essay + Save")

    filename_input = gr.Textbox(label="Filename to save (no extension)", placeholder="essay_01")
    status = gr.Markdown("")

    gen_btn.click(fn=handle_generate,
                  inputs=[mode_selector, user_input, output, followups],
                  outputs=[output, followups, status])

    clear_btn.click(fn=handle_clear,
                    outputs=[output, followups, status])

    finish_btn.click(fn=handle_finish,
                     inputs=[filename_input],
                     outputs=status)

if __name__ == "__main__":
    demo.launch()
