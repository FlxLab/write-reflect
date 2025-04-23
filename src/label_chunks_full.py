"""
label_chunks_full.py

This script uses the local LLaMA3:8B model via Ollama to classify each chunk of text
with 1–3 thematic tags from a fixed list, and extract the reasoning behind the classification.

Input (reads):
- 'writing_chunks.csv'

Output (writes):
- 'writing_chunks_labeled.csv' with columns:
  - id: chunk number
  - text: original text
  - tags: comma-separated list of selected tags
  - reasoning: model-generated explanation

Model: llama3:8b (Ollama)
"""

import csv
import requests
import time
import re


# Custom thematic tags list
tags = [
    "african_values_and_worldviews",
    "critique_of_western_ai_ethics",
    "language_and_translation",
    "infrastructure_and_data_realities",
    "youth_and_futures",
    "personal_reflection",
    "african_ai_startups_and_case_studies",
    "power_dynamics_global_north_south",
    "climate_and_frugal_innovation",
    "speculative_or_poetic_expression",
    "critical_question_or_problem_statement",
    "connective_or_transition_fragment"
]

tag_list_string = "; ".join(tags)

# Base prompt
prompt_template = f"""
Classify the following text based on the most relevant tags from this list:
{tag_list_string}
Return only a comma-separated list of tags first, followed by a short explanation of why you chose them.

Text:
"""

from pathlib import Path

# Define base directory dynamically (points one level up from /src)
base_path = Path(__file__).resolve().parent.parent
input_csv = base_path / "data" / "processed" / "writing_chunks.csv"
output_csv = base_path / "data" / "processed" / "writing_chunks_labeled.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

with open(input_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    chunks = list(reader)

results = []

for i, row in enumerate(chunks):
    chunk_id = row["id"]
    text = row["text"]

    if not text.strip():
        print(f"Skipping empty chunk {chunk_id}")
        continue

    prompt = prompt_template + text

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            result_text = response.json().get("response", "").strip()

            # Split into tags and reasoning
            lines = result_text.split("\n")
            # Find first line that contains likely tags (comma-separated, using our known tag keywords)
            tag_line_index = None
            for i, line in enumerate(lines):
                if "," in line and any(tag in line for tag in tags):
                    tag_line_index = i
                    break
            if tag_line_index is not None:
                 clean_tags = lines[tag_line_index].strip()
                 reasoning = "\n".join(lines[tag_line_index + 1:]).strip()
            else:
                 clean_tags = "ERROR"
                 reasoning = result_text
                 
            print(f"✓ Chunk {chunk_id} → Tags: {clean_tags}")
        else:
            print(f"✗ Error at chunk {chunk_id} (HTTP {response.status_code})")
            clean_tags = "ERROR"
            reasoning = ""

    except Exception as e:
        print(f"✗ Request failed at chunk {chunk_id}: {e}")
        clean_tags = "ERROR"
        reasoning = ""

    results.append({
        "id": chunk_id,
        "text": text,
        "tags": clean_tags,
        "reasoning": reasoning
    })

# Write to CSV
with open(output_csv, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text", "tags", "reasoning"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n Classification complete. Output saved to: {output_csv}")
