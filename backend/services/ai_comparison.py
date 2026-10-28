import json
import logging
import math
import time
from hashlib import sha256
from statistics import mean, median
from uuid import uuid4

from sqlalchemy import Float, String, Text, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.core.ai_config import settings
from backend.core.database import Base
from backend.schemas.ai import RouteDecision
from backend.services import ai_providers, ai_tracing

logger = logging.getLogger(__name__)
PROTOCOL = "router-comparison-v1"
DATASET = [
    ("sql", "Show all unique suppliers in inventory."),
    ("sql", "Which products have the maximum unit price?"),
    ("sql", "Show stock quantity for item 12."),
    ("sql", "Show inventory value grouped by category."),
    ("faq", "What does the operations manual say about negative stock?"),
    ("faq", "Who approves stock adjustments according to the documents?"),
    ("faq", "Summarize findings in the monthly inventory audit report."),
    ("faq", "What supplier controls does the procurement policy describe?"),
    ("quantity", "Set item 12 stock quantity to 80."),
    ("quantity", "Set item 12 to 80 units and item 7 to 25 units."),
    ("quantity", "Increase item 3 stock quantity to 10."),
    ("quantity", "Set item 8 stock quantity to zero."),
    ("price", "Set the price of product ID 379 from NULL to 120."),
    ("price", "Set item 12 price to 19.99 and item 7 price to 25."),
    ("price", "Increase the unit price of item 4 to 100."),
    ("price", "Set item 5 unit price to zero."),
    ("unsupported", "Delete item 12 from inventory."),
    ("unsupported", "Purchase 100 units from our supplier."),
    ("unsupported", "Increase every product price by ten percent."),
    ("unsupported", "Set item 12 quantity to 10 and its price to 20."),
]


class RouterComparison(Base):
    __tablename__ = "router_comparisons"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[float] = mapped_column(Float)
    mode: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    result: Mapped[str] = mapped_column(Text)


def create(database: Session, mode: str, run_id: str | None = None) -> dict:
    result = {
        "id": str(uuid4()), "created_at": time.time(), "mode": mode, "status": "running",
        "protocol": PROTOCOL, "dataset_version": "inventory-routing-20-v1" if mode == "benchmark" else None,
        "run_id": run_id, "total": len(DATASET) if mode == "benchmark" else 1, "pairs": [],
        "models": {name: settings().providers[name].model for name in ("jev", "mistral")},
        "trace_id": None, "error": None,
    }
    database.add(RouterComparison(
        id=result["id"], created_at=result["created_at"], mode=mode,
        status=result["status"], result=json.dumps(result),
    ))
    database.commit()
    return result


def save(database: Session, result: dict) -> None:
    record = database.get(RouterComparison, result["id"])
    if record is None:
        raise RuntimeError("Router comparison record missing")
    record.status = result["status"]
    record.result = json.dumps(result, allow_nan=False)
    database.commit()


def recover(database: Session) -> None:
    for record in database.scalars(select(RouterComparison).where(RouterComparison.status == "running")):
        result = json.loads(record.result)
        result.update(status="interrupted", error="Backend restarted; partial results retained. No evaluations are automatically retried.")
        save(database, result)


def evaluate(provider: str, message: str) -> tuple[dict, RouteDecision | None, ai_providers.ProviderResult | None]:
    decision = None
    response = None
    error = None
    with ai_tracing.span(f"compare_{provider}_router") as active:
        started = time.perf_counter()
        try:
            decision, response = (
                ai_providers.route(message) if provider == "jev" else ai_providers.route_mistral(message)
            )
        except ai_providers.ProviderError as exc:
            error = str(exc)
            logger.warning("Router comparison %s failed: %s", provider, type(exc).__name__)
            active.set_status("ERROR")
        latency_ms = (time.perf_counter() - started) * 1000
        if decision is not None and response is not None:
            ai_tracing.record_usage(active, response)
            active.set_attributes({"agent": decision.agent, "confidence": decision.confidence})
    return {
        "provider": provider, "model": response.model if response else settings().providers[provider].model,
        "status": "error" if error else "ok", "error": error,
        "latency_ms": latency_ms,
        "agent": decision.agent if decision else None,
        "confidence": decision.confidence if decision else None,
        "input_tokens": response.usage.input_tokens if response else None,
        "output_tokens": response.usage.output_tokens if response else None,
        "attempts": response.attempts if response else 1,
    }, decision, response


def pair(message: str, index: int, expected: str | None = None) -> dict:
    results = {}
    # Alternate which endpoint runs first, with sequential calls to avoid competition.
    order = ("jev", "mistral") if index % 2 == 0 else ("mistral", "jev")
    for provider in order:
        results[provider] = evaluate(provider, message)[0]
    return {
        "index": index, "expected": expected,
        "input_hash": sha256(message.encode()).hexdigest(), "order": list(order), **results,
    }


def benchmark(database: Session, result: dict) -> None:
    with ai_tracing.capture(), ai_tracing.span("router_benchmark", protocol=PROTOCOL) as active:
        result["trace_id"] = active.trace_id
        save(database, result)
        for index, (expected, message) in enumerate(DATASET):
            row = pair(message, index, expected)
            row["question"] = message
            result["pairs"].append(row)
            save(database, result)
        result["status"] = "completed"
        save(database, result)


def summarize(pairs: list[dict]) -> dict:
    paired = [row for row in pairs if all(row[p]["status"] == "ok" for p in ("jev", "mistral"))]
    token_pairs = [
        row for row in paired
        if all(row[p][key] is not None for p in ("jev", "mistral") for key in ("input_tokens", "output_tokens"))
    ]
    providers = {}
    for provider in ("jev", "mistral"):
        samples = [row[provider] for row in paired]
        latencies = sorted(row["latency_ms"] for row in samples)
        reported = [row[provider] for row in token_pairs]
        providers[provider] = {
            "attempted": len(pairs),
            "errors": sum(row[provider]["status"] == "error" for row in pairs),
            "paired_samples": len(samples),
            "mean_ms": mean(latencies) if latencies else None,
            "median_ms": median(latencies) if latencies else None,
            "p95_ms": latencies[math.ceil(len(latencies) * .95) - 1] if latencies else None,
            "token_samples": len(reported),
            "reported_token_samples": sum(
                row["input_tokens"] is not None and row["output_tokens"] is not None for row in samples
            ),
            "mean_input_tokens": mean(row["input_tokens"] for row in reported) if reported else None,
            "mean_output_tokens": mean(row["output_tokens"] for row in reported) if reported else None,
        }
    agreements = sum(row["jev"]["agent"] == row["mistral"]["agent"] for row in paired)
    return {
        "providers": providers, "paired_samples": len(paired), "agreements": agreements,
        "agreement_rate": agreements / len(paired) if paired else None,
    }


def view(result: dict) -> dict:
    return {**result, "completed": len(result["pairs"]), "metrics": summarize(result["pairs"])}
