import json
from pathlib import Path

from src.generation.demo_prompting import (
    load_model,
    load_retriever,
    retrieve_chunks,
    generate_answer,
)


def load_sample_qa():
    project_root = Path(__file__).resolve().parents[2]
    qa_path = project_root / "data" / "samples" / "mediterranean_sample_qa.json"

    with open(qa_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_questions(sample_qa):
    items = []

    for block in sample_qa["sources"]:
        source = block.get("source", "unknown")

        for qa in block.get("questions", []):
            items.append({
                "source": source,
                "question": qa["question"],
                "answer": qa["answer"],
            })

    return items

def main():
    tokenizer, model = load_model()
    retriever = load_retriever()

    sample_qa = load_sample_qa()

    qa_items = flatten_questions(sample_qa)
    print(f"Loaded {len(qa_items)} QA items")

    if qa_items:
        print("Example:", qa_items[0])


    for i, item in enumerate(qa_items[:5], start=1):
        question = item["question"]
        gold_answer = item["answer"]

        retrieved = retrieve_chunks(question, retriever)
        chunk_texts = [x["text"] for x in retrieved]

        strict_answer = generate_answer(
            question=question,
            chunks=chunk_texts,
            tokenizer=tokenizer,
            model=model,
            prompt_mode="strict"
        )

        baseline_answer = generate_answer(
            question=question,
            chunks=chunk_texts,
            tokenizer=tokenizer,
            model=model,
            prompt_mode="baseline"
        )

        print(f"\n===== QA {i} =====")
        print("Source:", item["source"])
        print("Question:", question)
        print("Gold answer:", gold_answer)
        print("Strict:", strict_answer)
        print("Baseline:", baseline_answer)

        print("\nRetrieved context:")
        for j, r in enumerate(retrieved, start=1):
            doc_id = r.get("metadata", {}).get("doc_id", r.get("chunk_id", f"chunk_{j:03d}"))
            print(f"[{j}] ({doc_id}) {r['text'][:180]}...")


if __name__ == "__main__":
    main()