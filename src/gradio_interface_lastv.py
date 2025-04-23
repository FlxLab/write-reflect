"""
gradio_interface.py

Interactive Gradio interface for your Critical AI Writing Companion.
Supports two modes:
1. Q&A Mode — Ask a reflective question and get a one-off response.
2. Essay Builder — Build an essay section-by-section and save it at the end.
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

# Load archive
data = load_archive("embedded_chunks.pkl")

# Essay content store
essay_sections = []
followups = []

# Mode switch
def toggle_mode(mode):
    if mode == "Essay Builder":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

# Essay section builder
def add_essay_section(query, current_text, current_fups):
    query_embedding = embed_query(query)
    top_chunks = get_top_chunks(query_embedding, data)
    prompt = format_chunks_as_context(top_chunks, query)
    result = query_llm(prompt)

    # Extract essay + follow-up questions
    if "--- FOLLOW-UP-BEGIN ---" in result:
        main, fup = result.split("--- FOLLOW-UP-BEGIN ---", 1)
        fup = fup.strip()
    else:
        main = result.strip()
        fup = ""

    section_text = f"[Prompt: {query}]\n{main.strip()}"
    essay_sections.append(section_text)
    followups.append(fup)

    combined = current_text.strip() + "\n\n---\n\n" + section_text.strip()

    # ✅ Fixed logic here
    if fup:
        combined_fups = current_fups.strip() + "\n\n--- Follow-up Questions ---\n" + fup.strip()
    else:
        combined_fups = current_fups.strip()

    return combined.strip(), "", gr.update(value=combined_fups.strip(), visible=bool(combined_fups.strip()))






# Final save logic
def undo_last():
    if essay_sections:
        essay_sections.pop()
    if followups:
        followups.pop()

    combined = "\n\n---\n\n".join(essay_sections)
    combined_fups = "\n\n--- Follow-up Questions ---\n".join(
        [fup.strip() for fup in followups if fup.strip()]
    )

    return (
        combined.strip(),
        "",  # clears input_essay
        gr.update(value=combined_fups, visible=bool(combined_fups)),
        "",  # clears save_folder
        ""   # clears save_name
    )


# Clear all essay content (reset everything)
def clear_all():
    essay_sections.clear()
    followups.clear()
    return "", "", "", "", "", ""


def save_essay(path, filename, filetype):
    if not filename.strip():
        return "❌ Filename missing."

    if not path.strip():
        path = os.getcwd()

    os.makedirs(path, exist_ok=True)
    essay_path = os.path.join(path, f"{filename}{filetype}")
    fup_path = os.path.join(path, f"{filename}_followups{filetype}")

    with open(essay_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(essay_sections))

    with open(fup_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(fup for fup in followups if fup.strip()))

    return f"✅ Essay saved to:\n{essay_path}\n✅ Follow-up questions saved to:\n{fup_path}"

# Q&A logic
def run_qa(query):
    query_embedding = embed_query(query)
    top_chunks = get_top_chunks(query_embedding, data)
    prompt = format_chunks_for_qa(top_chunks, query)
    return query_llm(prompt)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## ✍️ Critical AI Writing Companion")

    mode = gr.Radio(choices=["Essay Builder", "Q&A"], value="Essay Builder", label="Mode")

    # Essay Builder Section
    with gr.Column(visible=True) as essay_builder:
        input_essay = gr.Textbox(label="Your next essay prompt", placeholder="e.g. Why is decolonial AI important?")
        essay_display = gr.Textbox(label="Generated Essay", lines=20)
        followup_display = gr.Textbox(label="Follow-up Questions", lines=4, visible=False)


        add_btn = gr.Button("Add Section")
        clear_btn = gr.Button("Clear All")
        undo_btn = gr.Button("Undo Last Section")
        finish_btn = gr.Button("Finish Essay + Save")

        save_folder = gr.Textbox(label="📁 Folder path (optional)", placeholder="Leave blank for current folder")
        save_name = gr.Textbox(label="📝 Filename base (no extension)")
        save_format = gr.Radio(choices=[".txt", ".md"], value=".txt", label="File Format")
        save_status = gr.Textbox(label="Save Status", interactive=False)

        add_btn.click(add_essay_section,inputs=[input_essay, essay_display, followup_display],outputs=[essay_display, input_essay, followup_display])

        clear_btn.click(fn=clear_all, outputs=[essay_display, input_essay, followup_display, save_folder, save_name, save_status])
        undo_btn.click(fn=undo_last, outputs=[essay_display, input_essay, followup_display, save_folder, save_name])
        finish_btn.click(save_essay, inputs=[save_folder, save_name, save_format], outputs=save_status)


    # Q&A Section
    with gr.Column(visible=False) as qa_mode:
        qa_input = gr.Textbox(label="Ask your question")
        qa_output = gr.Textbox(label="Reflective Answer", lines=15)
        qa_button = gr.Button("Generate Answer")
        qa_button.click(run_qa, inputs=qa_input, outputs=qa_output)

    # 🔁 Now that essay_builder and qa_mode are defined, this line works:
    mode.change(fn=toggle_mode, inputs=mode, outputs=[essay_builder, qa_mode])


# Launch
if __name__ == "__main__":
    demo.launch()
