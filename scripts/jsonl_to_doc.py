import json
import os

def load_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_documents_as_markdown(dataset, output_dir="synthetic_dataset/markdown_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    for i, entry in enumerate(dataset):
        doc_text = entry.get("document")
        doc_id = entry.get("document_id", f"doc_{i+1}")
        if doc_text:
            filename = os.path.join(output_dir, f"document_{doc_id}.md")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(doc_text)
            print(f"Saved: {filename}")
        else:
            print(f"Skipping entry {i} (no document field)")

# Usage
dataset = load_jsonl("synthetic_dataset/datapoints.jsonl")
save_documents_as_markdown(dataset)


