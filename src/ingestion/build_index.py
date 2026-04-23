import json
import os
from pathlib import Path

from src.retrieval.embedder import TextEmbedder


def load_chunks(chunk_path: str):
    chunks = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def save_index(index_data, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)


def main():
    project_root = Path(__file__).resolve().parents[2]

    chunk_path = project_root / "data" / "processed" / "chunks_v3.jsonl"
    output_path = project_root / "data" /"retrieval" / "index_multiqa.json"

    print(f"Loading chunks from: {chunk_path}")
    chunks = load_chunks(str(chunk_path))
    print(f"Loaded {len(chunks)} chunks.")

    texts = [chunk["text"] for chunk in chunks]

    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    embedder = TextEmbedder(model_name=model_name)
    embeddings = embedder.embed_documents(texts)

    if not embeddings:
        print("No embeddings generated.")
        return

    index_data = []
    for chunk, emb in zip(chunks, embeddings):
        index_data.append({
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
            "metadata": chunk.get("metadata", {}),
            "embedding": emb
        })

    save_index(index_data, str(output_path))

    print(f"Saved index to: {output_path}")
    print(f"Number of indexed chunks: {len(index_data)}")
    print(f"Embedding dimension: {len(index_data[0]['embedding'])}")


if __name__ == "__main__":
    main()