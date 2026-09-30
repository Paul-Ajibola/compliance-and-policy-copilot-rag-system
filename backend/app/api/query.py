"""
The /query endpoint: runs the confidence gate first (cheap, no LLM call),
and only proceeds to generation if the gate is confident.
"""
import json
import logging
import traceback

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session
from app.confidence.gate import evaluate_query
from app.generation.generator import stream_answer, build_citations

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/query")
async def query(request: QueryRequest, session: AsyncSession = Depends(get_session)):
    async def event_stream():
        try:
            gate_result = await evaluate_query(session, request.question)
        except Exception as e:
            logger.error("Gate/retrieval failed:\n%s", traceback.format_exc())
            yield _sse_event("error", {"message": f"retrieval failed: {type(e).__name__}: {e}"})
            return

        if not gate_result.confident:
            yield _sse_event("refused", {
                "message": "I don't have enough confidence in the available policy documents to answer this question reliably. This has been escalated for human review.",
                "top_score": gate_result.top_score,
            })
            return

        citations = build_citations(gate_result.all_results)
        yield _sse_event("citations", {"citations": citations})

        try:
            async for token in stream_answer(request.question, gate_result.all_results):
                yield _sse_event("token", {"text": token})
        except Exception as e:
            logger.error("Generation failed:\n%s", traceback.format_exc())
            yield _sse_event("error", {"message": f"generation failed: {type(e).__name__}: {e}"})
            return

        yield _sse_event("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")