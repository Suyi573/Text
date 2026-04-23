import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========= 1. Load corpus =========
file_path = Path("data/processed/corpus.jsonl")
documents = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        documents.append(json.loads(line))

print(f" Successfully loaded {len(documents)} documents.\n")

# ========= 2. Configure chunker =========
# Only used for long blocks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "? ", "! ", "。", "！", "？", " ", ""]
)

# ========= 3. Process all documents =========
print("Starting chunking...")
final_chunks = []
chunk_id_counter = 0

for doc in documents:
    text = doc.get("text", "").strip()
    if not text:
        continue

    # Keep original metadata fields
    metadata = {k: v for k, v in doc.items() if k != "text"}

    # IMPORTANT:
    # If the cleaned corpus block is already short enough, keep it as one chunk.
    if len(text) <= 700:
        candidate_chunks = [text]
    else:
        candidate_chunks = splitter.split_text(text)

    for i, c_text in enumerate(candidate_chunks):
        c_text = c_text.strip()
        c_text = c_text.lstrip(".,;:!?，。；：！？、")
        c_text = c_text.strip()

        # stronger filtering
        if len(c_text) < 80:
            continue
        if len(c_text.split()) < 12:
            continue

        chunk_record = {
            "chunk_id": f"chunk_{chunk_id_counter}",
            "text": c_text,
            "metadata": {
                **metadata,
                "source_doc_id": doc.get("doc_id"),
                "chunk_index": i,
                "chunk_method": "keep_whole" if len(text) <= 700 else "recursive_split"
            }
        }

        final_chunks.append(chunk_record)
        chunk_id_counter += 1
# ========= 4. Deduplicate =========
print("Removing duplicate chunks...")

seen_texts = set()
deduped_chunks = []

for chunk in final_chunks:
    text = chunk["text"].strip()

    if text in seen_texts:
        continue

    seen_texts.add(text)
    deduped_chunks.append(chunk)

print(f"Removed {len(final_chunks) - len(deduped_chunks)} duplicates")

final_chunks = deduped_chunks

# ========= 4. Save =========
output_path = Path("data/processed/chunks_v3.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for chunk in final_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"Done! {len(documents)} documents were converted into {len(final_chunks)} chunks.")
print(f"Final data saved to: {output_path}")