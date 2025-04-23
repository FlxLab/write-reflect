# Write-Reflect: Local Writing Companion for Reflection & Retrieval

A local-first writing tool that helps you reflect on your own words. It turns your personal texts into a responsive co-thinker — not a chatbot or black-box oracle, but a critical conversation partner grounded in your own voice, values, and writing. Powered by semantic search, thematic annotation, and local LLMs (via Ollama), this tool lets you write, retrieve, and rethink from within — generating new insights, reflective essays, and follow-up questions drawn from your own writing.

## Conceptual Overview

This project explores how AI tools can support **reflective writing practices** by turning personal text archives into searchable knowledge bases. It fuses two distinct layers of language processing:

### 1. **Semantic NLP** – classification, embedding, retrieval
- Personal writing is split into chunks (~100–200 words), semantically classified with a local LLM, and embedded using `sentence-transformers`.
- This enables **vector-based retrieval**: each user query pulls thematically closest chunks from your own writing.

### 2. **Generative AI** – contextual re-writing with local LLMs
- Retrieved excerpts are passed to an LLM with custom prompts that:
  - **Synthesise ideas** without copying source text
  - Emphasise clarity, voice, and cultural grounding
  - Always return **follow-up questions** to deepen reflection

This double-layer structure supports both associative flow and critical rigour — and foregrounds **question-asking** over answer-giving.

## Features

| Capability                      | Description |
|----------------------------------|-------------|
| Reflective Q&A                  | Ask a question, get a grounded 1–2 paragraph response + follow-up questions |
| Custom Essay Builder            | Add one prompt at a time to build a structured multi-part essay |
| Thematic tagging                | Each text chunk includes topic tags and interpretive reasoning |
| Fully local + privacy-focused   | No cloud API required — runs via [Ollama](https://ollama.com) |
| Follow-up logic                 | Encourages curiosity, not closure |

## Tools & Technologies

| Tool                | Role |
|---------------------|------|
| Python              | Core scripting and orchestration |
| Ollama              | Runs local LLMs (`llama3`, `mistral`) for classification and generation |
| sentence-transformers | Embeds text for semantic search |
| pandas              | Handles dataset manipulation |
| Gradio              | Local web interface for interaction |
| VS Code             | Development environment |
| (planned) p5.js     | Creative, poetic frontend for Q&A mode |
| (optional) PyMuPDF  | To later embed PDF archives |

## Pipeline: How It Works

1. **Prepare Your Writing**
   - Combine personal texts into a single `.txt`
   - Segment into thematic chunks (`segment_text.py`)

2. **Annotate with LLM**
   - Use `label_chunks_full.py` to add **tags** and **reasoning** to each chunk

3. **Vectorise**
   - `embed_chunks.py` creates embeddings via `MiniLM-L6-v2`
   - Stores data in `embedded_chunks.pkl`

4. **Build or Explore**
   - `generate_response.py`: Ask reflective questions, get short answers
   - `generate_essay.py`: Build longform responses, prompt-by-prompt
   - `gradio_interface.py`: Choose between both modes in a web interface

## Modes of Use

### Q&A Mode
- Use: `generate_response.py` or Gradio UI
- Input: "What are some concrete examples of African-centered AI applications that prioritize collective well-being and social responsibility over individualistic gains?"
- Output: A thoughtful paragraph + 2–3 follow-up questions

### Essay Builder
- Use: `generate_essay.py` or Gradio UI
- Input: Prompts like "What strategies can be employed to ensure that diverse perspectives, especially those from marginalized communities, are included in the ethical discourse around AI?"
- Output: Full essay section-by-section + follow-up questions saved to `.txt`

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- [Ollama](https://ollama.com) for running local LLMs

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/FlxLab/write-reflect.git
cd write-reflect
```

2. **Set up a Python environment**
```bash
# Using conda
conda create -n write-reflect python=3.8
conda activate write-reflect

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install and start Ollama**
```bash
# macOS
brew install ollama

# Linux/Windows
# Follow instructions at https://ollama.com

# Start Ollama server
ollama serve

# Download and run the LLM model
ollama run llama3  # or mistral
```

### Running the Project

1. **Process your text**
```bash
# Place your writing in source_materials.txt
python src/segment_text.py
python src/label_chunks_full.py
python src/embed_chunks.py
```

2. **Launch the interface**
```bash
python src/gradio_interface.py
```

## License

MIT License  
Free to use, remix, and reflect.
