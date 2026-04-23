from typing import List


def format_context(chunks: List[str]) -> str:
    """Join top-k chunks into a numbered context block."""
    lines = []
    for i, ch in enumerate(chunks, start=1):
        ch = ch.strip().replace("\n", " ")
        lines.append(f"[{i}] {ch}")
    return "\n".join(lines)


BASELINE_TEMPLATE = """Answer the question based on the context below.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""


STRICT_TEMPLATE = """You are a question answering system.

Answer the question using ONLY the provided context.

STRICT RULES:
- Use ONLY information explicitly stated in the context.
- If the answer is not explicitly stated or directly supported by the context, output exactly: I don't know based on the provided context.
- Do NOT use outside knowledge, guess, or infer unsupported information.
- Minimal normalization is allowed only when it preserves the exact meaning of the context.

PROCESS:
- First identify the most relevant part of the context.
- Then check whether it contains a phrase or phrases that directly answer the question.
- If such a phrase exists, extract the shortest answer that clearly answers the question without losing necessary information.
- If no such phrase exists, output exactly: I don't know based on the provided context.

ANSWER RULES:
- Every answer must be directly grounded in the context.
- Do not output incomplete, truncated, or partial answers.
- Do not return unrelated words copied from the question or context.
- Do not return the topic word itself as the answer unless it directly answers the question.
- Prefer exact wording from the context, but allow minimal normalization when needed to directly answer the question.

QUESTION TYPE RULES:
- If the question asks "which", return only the specific item or items that answer the question.
- If multiple items are explicitly listed and all are needed, include all of them.
- If the question asks "what type", answer with the category or type only.
- If the question asks for a definition, extract the defining phrase, not the term alone.
- If the question asks about a difference, state the difference clearly using information from the context, including both sides.
- If the question asks "why", answer with the complete reason only.
- If the question asks "how", answer with the complete method or process only.

SAFETY CHECKS:
- If the question requires comparison, ranking, preference, or superlatives (for example: "most", "best", "more", "less", "preferred"), and this is not explicitly stated in the context, output exactly: I don't know based on the provided context.
- Do not include bullet points unless the question explicitly asks for a list.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

FALLBACK_TEMPLATE = """You are a question answering system.

Answer the question using ONLY the provided context.

Rules:
- Use only information explicitly stated in the context.
- Do not use outside knowledge.
- Do not guess if the answer is absent.
- If the context contains wording that answers the question, extract the closest matching phrase.
- Minimal normalization is allowed only when it preserves the exact meaning.
- Prefer answering over outputting I don't know when the context clearly contains the answer.
- Do not return the topic word alone unless it directly answers the question.
- Do not return incomplete or truncated answers.
- If the answer is still not present, output exactly: I don't know based on the provided context.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

RESCUE_TEMPLATE = """You are a question answering system.

Answer the question using ONLY the provided context.

Rules:
- The answer is present in the context if a directly relevant phrase exists.
- Extract the shortest phrase or phrases from the context that answer the question.
- Do not require a perfect full sentence.
- For item questions, return only the item or items.
- For reason or method questions, return the relevant reason or method phrase.
- For definition questions, return the defining phrase.
- If no relevant phrase exists, output exactly: I don't know based on the provided context.
- Do not use outside knowledge.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""


FEW_SHOT_TEMPLATE = """You are a question answering system.

Answer the question using ONLY the provided context.

RULES:
- Do NOT use any external knowledge.
- If the answer is not clearly supported by the context, output exactly:
  I don't know based on the provided context.
- Extract the answer directly from the context whenever possible.
- Prefer short, precise answers.

EXAMPLES

Context:
[1] Chermoula is a wet north African marinade, made of finely processed coriander, oil, cumin, onion, preserved lemons, chilli, salt and pepper.

Question:
What is chermoula?

Answer:
Chermoula is a wet north African marinade made of coriander, oil, cumin, onion, preserved lemons, chilli, salt and pepper.

Context:
[1] For instance, cream or garlic must never, ever be included in a traditional carbonara.

Question:
Which two ingredients are traditionally not used in authentic Italian carbonara?

Answer:
Cream and garlic.

Context:
[1] Dolma refers to stuffed dishes, while sarma refers to wrapped dishes.

Question:
What is the difference between dolma and sarma?

Answer:
Dolma refers to stuffed dishes, while sarma refers to wrapped dishes.

NOW ANSWER THE FOLLOWING

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""


def build_prompt(question: str, chunks: list[str], mode: str = "strict") -> str:
    context = "\n\n".join(chunks)

    if mode == "strict":
        return STRICT_TEMPLATE.format(question=question, context=context)
    if mode == "fallback":
        return FALLBACK_TEMPLATE.format(question=question, context=context)
    if mode == "rescue":
        return RESCUE_TEMPLATE.format(question=question, context=context)
    if mode == "baseline":
        return BASELINE_TEMPLATE.format(question=question, context=context)
    if mode == "few_shot":
        return FEW_SHOT_TEMPLATE.format(question=question, context=context)

    raise ValueError(f"Unknown prompt mode: {mode}")