"""
Wraps the Gemini API for streaming generation, constrained to the
retrieved passages via prompts.py. Kept separate from the confidence
gate and the API route: this module only knows how to talk to the LLM,
nothing about HTTP or retrieval.
"""
from google import genai
from app.config import settings
from app.generation.prompts import build_prompt
from groq import AsyncGroq


_gemini_client: genai.Client | None = None
_groq_client: AsyncGroq | None = None 



def get_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


def _clean(text: str) -> str:
    """Normalize model output so the frontend's [n] citation parser works."""
    return (
        text.replace("\u3010", "[")   # fullwidth left bracket
            .replace("\u3011", "]")   # fullwidth right bracket
            .replace("\u202f", " ")   # narrow no-break space
            .replace("**", "")        # UI doesn't render markdown bold
    )



async def stream_answer(question: str, context_chunks: list[dict]):
    """Yields text chunks as they're generated."""
    prompt = build_prompt(question, context_chunks)

    if settings.llm_provider == "groq":
        stream = await get_groq_client().chat.completions.create(
            model=settings.groq_generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield _clean(delta)
        return

    stream = await get_client().aio.models.generate_content_stream(
        model=settings.gemini_model,
        contents=prompt,
    )
    async for chunk in stream:
        if chunk.text:
            yield chunk.text


async def generate_answer_full(question: str, context_chunks: list[dict]) -> str:
    """
    Non-streaming version, used by the offline LLM-as-judge evaluation
    script (eval/llm_judge.py), which needs one complete answer string.
    """
    prompt = build_prompt(question, context_chunks)

    if settings.llm_provider == "groq":
        response = await get_groq_client().chat.completions.create(
            model=settings.groq_generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return _clean(response.choices[0].message.content or "")

    response = await get_client().aio.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )
    return response.text


def build_citations(context_chunks: list[dict]) -> list[dict]:
    """
    Maps citation numbers [1], [2], ... back to their source chunk
    metadata, for the frontend to render as clickable badges.
    """
    citations = []
    for i, chunk in enumerate(context_chunks, start=1):
        citations.append({
            "marker": i,
            "chunk_id": str(chunk.get("id")),
            "chunk_index": chunk.get("chunk_index"),
            "heading_path": chunk.get("heading_path"),
            "chunk_type": chunk.get("chunk_type"),
            "text": chunk.get("text"),
        })
    return citations