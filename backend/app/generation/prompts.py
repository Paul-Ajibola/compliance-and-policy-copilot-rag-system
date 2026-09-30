"""
Builds the system prompt that constrains generation to only use the
retrieved passages, with numbered citation markers the frontend (Phase 7)
can later render as clickable badges.
"""

SYSTEM_INSTRUCTIONS = """You are a compliance and policy assistant. Answer the user's question using ONLY the numbered passages provided below.

Rules:
- Every factual claim must be followed by a citation marker like [1] or [2] referencing the passage number it came from.
- If the passages do not contain enough information to answer the question, say so explicitly. Do not guess or use outside knowledge.
- Be concise and direct. Do not repeat the passages verbatim — synthesize them into a clear answer.
- Write plain text only, with no markdown formatting. Cite passages with ASCII square brackets exactly like [1], never with other bracket characters.
"""


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    passages = []
    for i, chunk in enumerate(context_chunks, start=1):
        heading = " > ".join(chunk.get("heading_path", []) or [])
        passages.append(f"[{i}] ({heading})\n{chunk['text']}")

    passages_block = "\n\n".join(passages)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"PASSAGES:\n{passages_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


