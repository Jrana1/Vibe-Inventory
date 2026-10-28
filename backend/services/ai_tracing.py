import os
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterator
from uuid import uuid4

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from backend.core.ai_config import ROOT, settings
from backend.services.ai_providers import ProviderError, ProviderResult

logger = logging.getLogger(__name__)


@dataclass
class TraceLog:
    stages: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    langsmith_run_id: str | None = None
    started: float = field(default_factory=time.perf_counter)


current_log: ContextVar[TraceLog | None] = ContextVar("ai_trace_log", default=None)
parent_run: ContextVar[str | None] = ContextVar("ai_parent_span", default=None)


@contextmanager
def capture() -> Iterator[TraceLog]:
    log = TraceLog()
    token = current_log.set(log)
    try:
        yield log
    finally:
        current_log.reset(token)


@lru_cache(maxsize=1)
def langsmith_client():
    from langsmith import Client
    from urllib3.util.retry import Retry

    return Client(
        api_url="https://api.smith.langchain.com", api_key=settings().langsmith_key,
        auto_batch_tracing=False, timeout_ms=2000, retry_config=Retry(total=0),
        hide_inputs=True, hide_outputs=True, omit_traced_runtime_info=True,
    )


def remote_call(method: str, **kwargs: Any) -> bool:
    from langsmith.utils import LangSmithError
    from requests.exceptions import RequestException

    try:
        getattr(langsmith_client(), method)(**kwargs)
        return True
    except (LangSmithError, RequestException) as exc:
        logger.warning("LangSmith %s failed: %s", method, type(exc).__name__)
        log = current_log.get()
        if log is not None and "LangSmith export failed; local traces are retained." not in log.warnings:
            log.warnings.append("LangSmith export failed; local traces are retained.")
        return False


class ObservedSpan:
    def __init__(self, active: Any, attributes: dict):
        self.active = active
        self.attributes = dict(attributes)
        self.status = "OK"

    @property
    def trace_id(self):
        return self.active.trace_id

    def set_attributes(self, attributes: dict) -> None:
        self.attributes.update(attributes)
        self.active.set_attributes(attributes)

    def set_status(self, status: str) -> None:
        self.status = status
        self.active.set_status(status)


def initialize() -> None:
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"
    import mlflow

    config = settings()
    if not config.tracking_uri.startswith(("sqlite:", "file:")):
        raise ValueError("This local showcase only supports local sqlite: or file: MLflow tracking")
    mlflow.set_tracking_uri(config.tracking_uri)
    if mlflow.get_experiment_by_name("vibe-inventory-agent") is None:
        mlflow.create_experiment(
            "vibe-inventory-agent", artifact_location=(ROOT / "mlartifacts").as_uri(),
        )
    mlflow.set_experiment("vibe-inventory-agent")


def shutdown() -> None:
    if langsmith_client.cache_info().currsize:
        langsmith_client().close(timeout=2)
        langsmith_client.cache_clear()


def record_usage(active: Any, result: ProviderResult) -> None:
    active.set_attributes({
        "model": result.model, "latency_ms": result.latency_ms, "attempts": result.attempts,
        **result.usage.model_dump(exclude_none=True),
    })


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    import mlflow

    # Never trace function arguments automatically: they can contain API keys or inventory.
    failure = None
    started = time.perf_counter()
    log = current_log.get()
    remote_id = str(uuid4())
    parent = parent_run.get()
    config = settings()
    exported = False
    if config.langsmith_enabled:
        if config.langsmith_key:
            exported = remote_call(
                "create_run", id=remote_id, name=name, run_type="chain", inputs={},
                project_name=config.langsmith_project, parent_run_id=parent,
                start_time=datetime.now(timezone.utc), extra={"metadata": attributes},
            )
            if parent is None and exported and log is not None:
                log.langsmith_run_id = remote_id
        else:
            logger.warning("LangSmith enabled but LANGSMITH_API_KEY is missing")
            if log is not None and "LangSmith key is missing." not in log.warnings:
                log.warnings.append("LangSmith key is missing.")
    parent_token = parent_run.set(remote_id if exported else None)
    with mlflow.start_span(name=name, attributes=attributes) as active:
        observed = ObservedSpan(active, attributes)
        try:
            yield observed
        except GeneratorExit as exc:
            observed.set_status("ERROR")
            observed.set_attributes({"error_type": "Interrupted"})
            failure = exc
        except (HTTPException, ValidationError, ValueError, ProviderError, SQLAlchemyError) as exc:
            # Validation/SQL exceptions can embed input values; keep only their type in MLflow.
            observed.set_status("ERROR")
            observed.set_attributes({"error_type": type(exc).__name__})
            failure = exc
        finally:
            duration = (time.perf_counter() - started) * 1000
            if log is not None:
                log.stages.append({
                    "name": name, "offset_ms": (started - log.started) * 1000,
                    "duration_ms": duration, "status": observed.status,
                    "attributes": observed.attributes,
                })
            if exported:
                remote_call(
                    "update_run", run_id=remote_id, end_time=datetime.now(timezone.utc),
                    outputs={}, error="Stage failed" if observed.status == "ERROR" else None,
                    extra={"metadata": {**observed.attributes, "duration_ms": duration}},
                )
            parent_run.reset(parent_token)
    if failure is not None:
        raise failure
