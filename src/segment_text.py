"""
segment_text.py

This script reads a long .txt file of personal writing and segments it into manageable chunks
(e.g. short paragraphs or grouped sentences) to prepare for classification or embedding.

Input:
- 'my_writing_ai_africa.txt' (raw unstructured writing)

Output:
- 'writing_chunks.csv' with:
  - id: numeric ID of the chunk
  - text: the segmented unit of writing

Segmentation logic:
- Splits by double newlines, filters empty results
- Strips extra whitespace and normalises line breaks

This prepares text for use in NLP tasks like classification, summarisation, or vector embedding.
"""

import re
import os

from pathlib import Path

# Set up file paths
base_path = Path("/Users/flurina/NLPProjects/ai-africa-essay-tool")
input_path = base_path / "my_writing_ai_africa.txt"
output_path = base_path / "writing_chunks.csv"

# Load the text
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Split by paragraphs (double newlines), splits text wherever there are two empty lines
raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

# Segment long paragraphs into smaller chunks (~1–3 sentences each)
def chunk_paragraph(paragraph, max_sentences=3):
    # Naive sentence splitting (could later use spaCy if needed)
    # splits a paragraph into individual sentences using punctuation (period, exclamation, question mark) as the split point
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    #group sentences together into chunks of ~3 sentences; so a paragraph with 9 sentences becomes 3 smaller chunks;keeps each chunk focused but still meaningful
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

# Process all paragraphs
all_chunks = []
for para in raw_paragraphs:
    chunks = chunk_paragraph(para)
    all_chunks.extend(chunks)

# Chunked data gets written and saved to a CSV file
import csv

# Open a new CSV file to write the data
with open(output_path, "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)

    # Write the header row: column names ; creates the column headers for the CSV file
    writer.writerow(["id", "text"])

      # For each chunk of text, write a row with a unique ID
    for i, chunk in enumerate(all_chunks): #loops over each text chunk, assigning an incremental ID (i) to each one.
        writer.writerow([i, chunk]) # writes a new row in the CSV where: i is the numeric ID; chunk is your 1–3 sentence text block


print(f"Done. {len(all_chunks)} chunks written to {output_path}")
