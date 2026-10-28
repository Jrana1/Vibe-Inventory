import json
import logging
from queue import Empty, Queue
from threading import BoundedSemaphore, Event, Thread
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.core.ai_config import settings
from backend.core.database import get_db
from backend.schemas.ai import ChatRequest
from backend.services import ai_agent, ai_comparison, ai_rag, ai_store, ai_tracing

router = APIRouter(prefix="/api/ai", tags=["inventory-copilot"])
Database = Annotated[Session, Depends(get_db)]
gate = BoundedSemaphore(1)
logger = logging.getLogger(__name__)


def require_local(request: Request) -> None:
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(403, "AI showcase is restricted to loopback clients; do not expose through a proxy")
    if request.url.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise HTTPException(403, "AI showcase requires a localhost host")
    origin = request.headers.get("origin")
    if origin and origin not in (
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ):
        raise HTTPException(403, "Untrusted browser origin")


def require_enabled() -> None:
    if not settings().enabled:
        raise HTTPException(503, "AI is disabled. Configure .env and set AI_ENABLED=true, then restart")


@router.get("/status", dependencies=[Depends(require_local)])
def status() -> dict:
    config = settings()
    return {
        "enabled": config.enabled, "scope": "local-showcase",
        "confidence_threshold": config.threshold,
        "knowledge": {
            "configured": bool(config.qdrant_key and config.qdrant_url),
            "embedding_model": config.embedding_model,
            "documents": [path.name for path in sorted(ai_rag.DOCUMENTS.glob("*.pdf"))],
        },
        "observability": {
            "mlflow": True, "langsmith_enabled": config.langsmith_enabled,
            "langsmith_configured": bool(config.langsmith_key),
        },
        "providers": [
            {"name": p.name, "model": p.model, "configured": bool(p.key)}
            for p in config.providers.values()
        ],
    }


@router.post("/chat", dependencies=[Depends(require_local), Depends(require_enabled)])
def chat(body: ChatRequest, database: Database) -> dict:
    if not gate.acquire(blocking=False):
        raise HTTPException(429, "Another AI operation is running; retry after it finishes")
    try:
        return ai_agent.chat(
            database, body.message, parent_run_id=str(body.parent_run_id) if body.parent_run_id else None,
            compare_routing=body.compare_routing,
        )
    finally:
        gate.release()


@router.post("/chat/stream", dependencies=[Depends(require_local), Depends(require_enabled)])
def stream_chat(body: ChatRequest, database: Database) -> StreamingResponse:
    if not gate.acquire(blocking=False):
        raise HTTPException(429, "Another AI operation is running; retry after it finishes")
    messages: Queue[str | None] = Queue()
    stopped = Event()
    bind = database.get_bind()

    def produce():
        try:
            # FastAPI 0.115 closes dependency sessions before streaming starts.
            with Session(bind=bind) as session:
                events = ai_agent.chat_events(
                    session, body.message, parent_run_id=str(body.parent_run_id) if body.parent_run_id else None,
                    compare_routing=body.compare_routing,
                )
                try:
                    for event in events:
                        if stopped.is_set():
                            break
                        messages.put(
                            f"event: {event['event']}\ndata: {json.dumps(event['data'], allow_nan=False)}\n\n"
                        )
                finally:
                    events.close()
        except (SQLAlchemyError, RuntimeError, ValueError) as exc:
            logger.error("AI streaming operation failed: %s", type(exc).__name__)
            messages.put('event: error\ndata: {"message":"The AI operation failed; check saved requests before retrying."}\n\n')
        finally:
            messages.put(None)
            gate.release()

    async def consume():
        try:
            while True:
                try:
                    message = await run_in_threadpool(messages.get, True, 1)
                except Empty:
                    yield ": keepalive\n\n"
                    continue
                if message is None:
                    break
                yield message
        finally:
            stopped.set()

    worker = Thread(target=produce, name="inventory-ai-stream", daemon=True)
    try:
        worker.start()
    except RuntimeError:
        gate.release()
        raise
    return StreamingResponse(
        consume(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs", dependencies=[Depends(require_local), Depends(require_enabled)])
def runs(database: Database) -> list[dict]:
    records = database.scalars(
        select(ai_store.AgentRun).where(
            func.json_extract(ai_store.AgentRun.result, "$.schema_version") == 2
        ).order_by(ai_store.AgentRun.created_at.desc()).limit(50)
    ).all()
    return [
        {
            "id": row.id, "created_at": row.created_at, "status": row.status,
            "agent": data.get("agent"), "latency_ms": data.get("latency_ms", 0),
            "explanation_status": data.get("explanation_status"),
            "usage": data.get("usage", []),
            "stages": [
                {"name": stage["name"], "duration_ms": stage["duration_ms"], "status": stage["status"]}
                for stage in data.get("stages", [])
                if stage["name"] in ("jev_router", "sql_agent", "faq_agent", "quantity_agent", "price_agent")
            ],
        }
        for row in records if (data := json.loads(row.result)).get("schema_version") == 2
    ]


@router.get("/comparisons", dependencies=[Depends(require_local), Depends(require_enabled)])
def comparisons(database: Database) -> dict:
    results = [
        json.loads(record.result)
        for mode, limit in (("live", 50), ("benchmark", 10))
        for record in database.scalars(
            select(ai_comparison.RouterComparison).where(ai_comparison.RouterComparison.mode == mode)
            .order_by(ai_comparison.RouterComparison.created_at.desc()).limit(limit)
        )
    ]
    live = [pair for result in results if result["mode"] == "live" for pair in result["pairs"]]
    return {
        "live": ai_comparison.summarize(live),
        "benchmarks": [ai_comparison.view(result) for result in results if result["mode"] == "benchmark"],
        "live_results": [ai_comparison.view(result) for result in results if result["mode"] == "live"],
    }


@router.post("/comparisons/benchmark", status_code=202, dependencies=[Depends(require_local), Depends(require_enabled)])
def start_benchmark(database: Database) -> dict:
    if not gate.acquire(blocking=False):
        raise HTTPException(429, "Another AI operation is running; retry after it finishes")
    result = None
    try:
        config = settings()
        if not all(config.providers[name].key for name in ("jev", "mistral")):
            raise HTTPException(503, "Both Jev and Mistral API keys are required for comparison")
        result = ai_comparison.create(database, "benchmark")
        response = ai_comparison.view(json.loads(json.dumps(result)))
        bind = database.get_bind()

        def produce():
            try:
                with Session(bind=bind) as session:
                    try:
                        ai_comparison.benchmark(session, result)
                    except (SQLAlchemyError, RuntimeError, ValueError) as exc:
                        session.rollback()
                        logger.error("Router benchmark failed: %s", type(exc).__name__)
                        result.update(status="failed", error="Benchmark interrupted by a backend error; partial results retained.")
                        ai_comparison.save(session, result)
            finally:
                gate.release()

        Thread(target=produce, name="inventory-router-benchmark", daemon=True).start()
    except (HTTPException, SQLAlchemyError, RuntimeError, ValueError) as exc:
        try:
            database.rollback()
            if result is not None:
                result.update(status="failed", error="Benchmark worker could not start; no evaluations are automatically retried.")
                ai_comparison.save(database, result)
            logger.error("Router benchmark startup failed: %s", type(exc).__name__)
        finally:
            gate.release()
        raise
    return response


@router.get("/comparisons/{comparison_id}", dependencies=[Depends(require_local), Depends(require_enabled)])
def comparison(comparison_id: str, database: Database) -> dict:
    record = database.get(ai_comparison.RouterComparison, comparison_id)
    if record is None:
        raise HTTPException(404, "Router comparison not found")
    return ai_comparison.view(json.loads(record.result))


@router.get("/runs/{run_id}", dependencies=[Depends(require_local), Depends(require_enabled)])
def run(run_id: str, database: Database) -> dict:
    row = database.get(ai_store.AgentRun, run_id)
    if row is None:
        raise HTTPException(404, "Agent run not found")
    result = json.loads(row.result)
    if result.get("schema_version") != 2:
        raise HTTPException(410, "Run belongs to the removed AI workflow")
    proposals = database.scalars(
        select(ai_store.AgentProposal).where(ai_store.AgentProposal.run_id == run_id)
    ).all()
    result["proposals"] = [ai_store.proposal_view(proposal) for proposal in proposals]
    return result


@router.post("/proposals/{proposal_id}/{decision}", dependencies=[Depends(require_local), Depends(require_enabled)])
def decide(proposal_id: str, decision: str, database: Database) -> dict:
    if decision not in ("approve", "reject"):
        raise HTTPException(422, "Decision must be approve or reject")
    with ai_tracing.span("proposal_decision", proposal_id=proposal_id, decision=decision) as active:
        return ai_store.decide(database, proposal_id, decision == "approve", active.trace_id)
