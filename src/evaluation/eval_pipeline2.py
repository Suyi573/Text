import json
import re
import os
from collections import Counter, defaultdict


def normalize(text):
    """
    Normalize text for fairer comparison.
    """
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return normalize(text).split()


def is_unanswerable_response(text):
    """
    Detect abstain / IDK-style answers.
    """
    text = normalize(text)

    patterns = [
        "i dont know based on provided context",
        "i dont know based on the provided context",
        "i dont know",
        "not mentioned",
        "unknown",
        "no information",
        "not enough information",
        "cannot be determined",
        "not provided in context",
        "not provided in the context",
        "insufficient information",
    ]
    return any(p in text for p in patterns)


def exact_match(pred, gold):
    return normalize(pred) == normalize(gold)


def keyword_match(pred, gold):
    """
    Your original metric, slightly stabilized.
    Useful as a soft rule-based metric.
    """
    pred = normalize(pred)
    gold = normalize(gold)

    if not gold:
        return False


    if len(gold.split()) <= 4:
        return gold in pred


    parts = re.split(r",| and | or ", gold)
    parts = [p.strip() for p in parts if len(p.strip()) > 2]

    if not parts:
        return gold in pred

    hits = sum(1 for p in parts if p in pred)
    return hits >= max(1, len(parts) // 2)


def token_f1(pred, gold):
    """
    Standard token overlap F1.
    """
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def get_question_type(question, answerable):
    """
    Very simple heuristic question typing for analysis.
    """
    if not answerable:
        return "unanswerable"

    q = (question or "").lower().strip()

    if q.startswith("why"):
        return "why"
    if q.startswith("how"):
        return "how"
    if "difference" in q or "compare" in q or "compared" in q or "contrast" in q:
        return "comparison"
    if q.startswith("what is") or q.startswith("who is") or q.startswith("what are"):
        return "definition"
    if q.startswith("which"):
        return "which"
    return "factoid"


def safe_div(a, b):
    return a / b if b else 0.0


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

benchmark_path = os.path.join(BASE_DIR, "data", "benchmark", "benchmark.json")
output_path = os.path.join(BASE_DIR, "data", "output", "output_payload.json")

with open(benchmark_path, "r", encoding="utf-8") as f:
    benchmark = {item["query_id"]: item for item in json.load(f)["items"]}

with open(output_path, "r", encoding="utf-8") as f:
    outputs = json.load(f)["results"]


# -------------------------
# Overall counters
# -------------------------
exact_correct = 0
keyword_correct = 0
total_f1 = 0.0
answerable_total = 0

unanswerable_correct = 0
unanswerable_total = 0

# IDK classification metrics
idk_tp = 0   # truly unanswerable, predicted IDK
idk_fp = 0   # answerable, but predicted IDK
idk_fn = 0   # unanswerable, but did not predict IDK

# Error analysis storage
error_cases = []

# Per-type stats
type_stats = defaultdict(lambda: {
    "count": 0,
    "exact_correct": 0,
    "keyword_correct": 0,
    "total_f1": 0.0,
    "idk_count": 0,
})


for item in outputs:
    qid = item["query_id"]
    pred = item.get("response", "")
    bench_item = benchmark[qid]

    gold = bench_item["gold_standard_answer"]
    answerable = bench_item["answerable"]
    query = item.get("query", bench_item.get("query", ""))
    qtype = get_question_type(query, answerable)

    pred_is_idk = is_unanswerable_response(pred)

    type_stats[qtype]["count"] += 1
    if pred_is_idk:
        type_stats[qtype]["idk_count"] += 1

    if answerable:
        answerable_total += 1

        em = exact_match(pred, gold)
        km = keyword_match(pred, gold)
        f1 = token_f1(pred, gold)

        if em:
            exact_correct += 1
            type_stats[qtype]["exact_correct"] += 1

        if km:
            keyword_correct += 1
            type_stats[qtype]["keyword_correct"] += 1

        total_f1 += f1
        type_stats[qtype]["total_f1"] += f1

        # IDK metrics
        if pred_is_idk:
            idk_fp += 1
            label = "over_abstain"
        elif em:
            label = "correct"
        elif f1 >= 0.5:
            label = "partial_match"
        else:
            label = "missed_answer"

    else:
        unanswerable_total += 1

        if pred_is_idk:
            unanswerable_correct += 1
            idk_tp += 1
            label = "correct_unanswerable"
        else:
            idk_fn += 1
            label = "failed_to_abstain"

    error_cases.append({
        "query_id": qid,
        "query": query,
        "question_type": qtype,
        "answerable": answerable,
        "gold": gold,
        "pred": pred,
        "label": label
    })


# -------------------------
# Overall metrics
# -------------------------
exact_acc = safe_div(exact_correct, answerable_total)
keyword_acc = safe_div(keyword_correct, answerable_total)
avg_f1 = safe_div(total_f1, answerable_total)
unanswerable_acc = safe_div(unanswerable_correct, unanswerable_total)

idk_precision = safe_div(idk_tp, idk_tp + idk_fp)
idk_recall = safe_div(idk_tp, idk_tp + idk_fn)
idk_f1 = safe_div(2 * idk_precision * idk_recall, idk_precision + idk_recall)


# -------------------------
# Per-question-type metrics
# -------------------------
type_summary = {}
for qtype, stats in type_stats.items():
    count = stats["count"]

    if qtype == "unanswerable":
        type_summary[qtype] = {
            "count": count,
            "unanswerable_accuracy": safe_div(unanswerable_correct, unanswerable_total) if unanswerable_total else 0.0,
            "idk_rate": safe_div(stats["idk_count"], count),
        }
    else:
        type_summary[qtype] = {
            "count": count,
            "exact_match_accuracy": safe_div(stats["exact_correct"], count),
            "keyword_match_accuracy": safe_div(stats["keyword_correct"], count),
            "average_token_f1": safe_div(stats["total_f1"], count),
            "idk_rate": safe_div(stats["idk_count"], count),
        }


# -------------------------
# Save reports
# -------------------------
summary = {
    "answerable_total": answerable_total,
    "unanswerable_total": unanswerable_total,
    "exact_match_accuracy": round(exact_acc, 4),
    "keyword_match_accuracy": round(keyword_acc, 4),
    "average_token_f1": round(avg_f1, 4),
    "unanswerable_accuracy": round(unanswerable_acc, 4),
    "idk_precision": round(idk_precision, 4),
    "idk_recall": round(idk_recall, 4),
    "idk_f1": round(idk_f1, 4),
}

summary_path = os.path.join(BASE_DIR, "data", "eval", "evaluation_summary.json")
errors_path = os.path.join(BASE_DIR, "data", "eval", "evaluation_error_cases.json")
type_path = os.path.join(BASE_DIR, "data", "eval", "evaluation_by_type.json")

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

with open(errors_path, "w", encoding="utf-8") as f:
    json.dump(error_cases, f, indent=2, ensure_ascii=False)

with open(type_path, "w", encoding="utf-8") as f:
    json.dump(type_summary, f, indent=2, ensure_ascii=False)


# -------------------------
# Print results
# -------------------------
print("=== Generation Evaluation ===")
print(f"Answerable Questions: {answerable_total}")
print(f"Unanswerable Questions: {unanswerable_total}")
print(f"Exact Match Accuracy: {exact_acc:.2f}")
print(f"Keyword Match Accuracy: {keyword_acc:.2f}")
print(f"Average Token F1: {avg_f1:.2f}")
print(f"Unanswerable Accuracy: {unanswerable_acc:.2f}")
print(f"IDK Precision: {idk_precision:.2f}")
print(f"IDK Recall: {idk_recall:.2f}")
print(f"IDK F1: {idk_f1:.2f}")

print("\n=== By Question Type ===")
for qtype, stats in type_summary.items():
    print(f"\n[{qtype}]")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

print("\nSaved files:")
print(f"- {summary_path}")
print(f"- {errors_path}")
print(f"- {type_path}")