import json
from pathlib import Path
from collections import defaultdict

from src.retrieval.retriever import Retriever


def hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k):
    top_k_ids = retrieved_chunk_ids[:k]
    return int(any(chunk_id in relevant_chunk_ids for chunk_id in top_k_ids))


def init_bucket():
    return {
        "count": 0,
        "hit1": 0,
        "hit3": 0,
        "hit5": 0,
    }


def update_bucket(bucket, h1, h3, h5):
    bucket["count"] += 1
    bucket["hit1"] += h1
    bucket["hit3"] += h3
    bucket["hit5"] += h5


def summarize_bucket(bucket):
    count = bucket["count"]
    if count == 0:
        return {
            "count": 0,
            "hit@1": 0.0,
            "hit@3": 0.0,
            "hit@5": 0.0,
        }

    return {
        "count": count,
        "hit@1": bucket["hit1"] / count,
        "hit@3": bucket["hit3"] / count,
        "hit@5": bucket["hit5"] / count,
    }


def print_summary(name, summary):
    print(f"\n[{name}]")
    print(f"Count: {summary['count']}")
    print(f"Hit@1: {summary['hit@1']:.3f}")
    print(f"Hit@3: {summary['hit@3']:.3f}")
    print(f"Hit@5: {summary['hit@5']:.3f}")


def main():
    project_root = Path(__file__).resolve().parents[2]

    benchmark_path = project_root / "data" / "benchmark" / "benchmark.json"
    index_path = project_root / "data" /"retrieval"/  "index_multiqa.json"
    output_path = project_root / "data" /"eval"/  "retrieval_eval_results.json"

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    retriever = Retriever(
    str(index_path),
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
    items = benchmark["items"]

    overall_bucket = init_bucket()
    answerable_bucket = init_bucket()
    difficulty_buckets = defaultdict(init_bucket)

    detailed_results = []

    for item in items:
        query = item["query"]
        query_id = item.get("query_id")
        answerable = item.get("answerable", True)
        difficulty = item.get("difficulty", "unknown")
        qtype = item.get("type", "unknown")
        source_category = item.get("source_category", "unknown")
        relevant_chunk_ids = item.get("relevant_chunk_ids", [])

        results = retriever.retrieve_with_rerank(query, top_k=5, lexical_weight=0.2)
        retrieved_chunk_ids = [r["chunk_id"] for r in results]

        h1 = hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, 1)
        h3 = hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, 3)
        h5 = hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, 5)

        # overall
        update_bucket(overall_bucket, h1, h3, h5)

        # answerable only
        if answerable:
            update_bucket(answerable_bucket, h1, h3, h5)

        # difficulty buckets
        update_bucket(difficulty_buckets[difficulty], h1, h3, h5)

        detailed_results.append({
            "query_id": query_id,
            "query": query,
            "answerable": answerable,
            "difficulty": difficulty,
            "type": qtype,
            "source_category": source_category,
            "relevant_chunk_ids": relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_results": results,
            "hit@1": h1,
            "hit@3": h3,
            "hit@5": h5
        })

    overall_summary = summarize_bucket(overall_bucket)
    answerable_summary = summarize_bucket(answerable_bucket)

    difficulty_summary = {
        level: summarize_bucket(bucket)
        for level, bucket in difficulty_buckets.items()
    }

    final_output = {
        "overall": overall_summary,
        "answerable_only": answerable_summary,
        "by_difficulty": difficulty_summary,
        "details": detailed_results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print_summary("Overall", overall_summary)
    print_summary("Answerable Only", answerable_summary)

    for level in ["easy", "medium", "hard", "negative"]:
        if level in difficulty_summary:
            print_summary(f"Difficulty = {level}", difficulty_summary[level])

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()