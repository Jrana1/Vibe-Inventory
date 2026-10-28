import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class RoutingSettings(BaseModel):
    confidence_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    key: str
    url: str


@dataclass(frozen=True)
class AISettings:
    enabled: bool
    threshold: float
    providers: dict[str, ProviderConfig]
    tracking_uri: str
    langsmith_enabled: bool = False
    langsmith_key: str = ""
    langsmith_project: str = "vibe-inventory"
    qdrant_url: str = ""
    qdrant_key: str = ""
    qdrant_alias: str = "vibe-inventory-faq"
    embedding_model: str = "mistral-embed"


@lru_cache
def settings() -> AISettings:
    # Optional AI dependencies are only required when explicitly enabled.
    if (ROOT / ".env").exists():
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    enabled = os.getenv("AI_ENABLED", "false").lower() == "true"
    threshold = RoutingSettings(
        confidence_threshold=float(os.getenv("JEV_CONFIDENCE_THRESHOLD", "0.8"))
    ).confidence_threshold
    providers = {}
    for name, key, model, url in [
        ("jev", "TYPESAFE_API_KEY", "jev-1.14.0", "https://api.typesafe.ai/v1/systemone"),
        ("mistral", "MISTRAL_API_KEY", "mistral-large-4-0", "https://api.mistral.ai/v1/chat/completions"),
    ]:
        providers[name] = ProviderConfig(
            name, os.getenv(f"{name.upper()}_MODEL", model), os.getenv(key, ""),
            url,
        )
    return AISettings(
        enabled, threshold, providers,
        os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"),
        os.getenv("LANGSMITH_ENABLED", "false").lower() == "true",
        os.getenv("LANGSMITH_API_KEY", ""),
        os.getenv("LANGSMITH_PROJECT", "vibe-inventory"),
        os.getenv("QDRANT_URL", "").rstrip("/"),
        os.getenv("QDRANT_API_KEY", ""),
        os.getenv("QDRANT_COLLECTION_ALIAS", "vibe-inventory-faq"),
        os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
    )
