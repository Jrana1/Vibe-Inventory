import hashlib
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx

from backend.core.ai_config import ROOT, ProviderConfig, settings
from backend.services import ai_providers, ai_tracing

logger = logging.getLogger(__name__)
DOCUMENTS = ROOT / "backend" / "faq_data"
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200
DIMENSIONS = 1024
SCORE_THRESHOLD = .65


class RAGError(ai_providers.ProviderError):
    pass


@dataclass(frozen=True)
class Corpus:
    version: str
    chunks: list[dict]
    documents: list[dict]


def corpus() -> Corpus:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    paths = sorted(DOCUMENTS.glob("*.pdf"))
    if not paths:
        raise RAGError("No PDF documents found in backend/faq_data")
    fingerprint = hashlib.sha256(
        f"pdf-rag-v1:{settings().embedding_model}:{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode()
    )
    chunks = []
    documents = []
    try:
        for path in paths:
            if path.stat().st_size > 20_000_000:
                raise RAGError("A knowledge PDF exceeds the 20 MB document limit")
            fingerprint.update(path.name.encode())
            fingerprint.update(path.read_bytes())
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise RAGError("Encrypted knowledge PDFs are not supported")
            if len(reader.pages) > 200:
                raise RAGError("A knowledge PDF exceeds the 200-page limit")
            before = len(chunks)
            for page_number, page in enumerate(reader.pages, 1):
                text = " ".join((page.extract_text() or "").split())
                if not text:
                    logger.warning("Knowledge PDF %s page %s has no extracted text; OCR is not enabled", path.name, page_number)
                    continue
                for offset in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = text[offset:offset + CHUNK_SIZE]
                    if chunk:
                        chunks.append({
                            "document": path.name, "page": page_number,
                            "offset": offset, "text": chunk,
                        })
                    if offset + CHUNK_SIZE >= len(text):
                        break
            if len(chunks) == before:
                raise RAGError(f"No readable text in {path.name}; OCR is required before indexing")
            documents.append({"document": path.name, "pages": len(reader.pages), "chunks": len(chunks) - before})
            if len(chunks) > 2000:
                raise RAGError("PDF knowledge exceeds the 2,000-chunk limit")
    except (OSError, PyPdfError, ValueError) as exc:
        logger.warning("PDF extraction failed: %s", type(exc).__name__)
        raise RAGError("A knowledge PDF could not be read; check the files and server logs") from exc
    version = fingerprint.hexdigest()
    for chunk in chunks:
        chunk["id"] = str(uuid5(NAMESPACE_URL, f"{version}:{chunk['document']}:{chunk['page']}:{chunk['offset']}"))
        chunk["corpus_version"] = version
        chunk["embedding_model"] = settings().embedding_model
    return Corpus(version, chunks, documents)


def qdrant(method: str, path: str, body: dict | None = None) -> dict:
    config = settings()
    url = urlparse(config.qdrant_url)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in ("", "/")):
        raise RAGError("Configure QDRANT_URL as an HTTPS cluster endpoint")
    if not config.qdrant_key:
        raise RAGError("QDRANT_API_KEY is not configured; add a fresh key to local .env")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", config.qdrant_alias):
        raise RAGError("QDRANT_COLLECTION_ALIAS must contain only letters, digits, hyphens or underscores")
    try:
        with httpx.Client(timeout=30, follow_redirects=False) as client:
            response = client.request(
                method, config.qdrant_url + path, json=body,
                headers={"api-key": config.qdrant_key},
            )
        if response.status_code not in (200, 201):
            logger.warning("Qdrant operation failed: HTTP %s", response.status_code)
            raise RAGError(f"Qdrant returned HTTP {response.status_code}; check key, cluster and indexing")
        payload = response.json()
        if not isinstance(payload, dict) or "result" not in payload:
            raise ValueError("Invalid Qdrant response")
        return payload
    except httpx.HTTPError as exc:
        logger.warning("Qdrant transport failed: %s", type(exc).__name__)
        raise RAGError("Qdrant connection failed or timed out") from exc
    except (ValueError, TypeError) as exc:
        logger.warning("Qdrant malformed response: %s", type(exc).__name__)
        raise RAGError("Qdrant returned an invalid response") from exc


def embed(texts: list[str]) -> tuple[list[list[float]], ai_providers.ProviderResult]:
    mistral = settings().providers["mistral"]
    config = ProviderConfig(
        "mistral", settings().embedding_model, mistral.key, "https://api.mistral.ai/v1/embeddings",
    )
    result = ai_providers.request(config, {"model": config.model, "input": texts, "encoding_format": "float"})
    try:
        data = sorted(result.payload["data"], key=lambda item: item["index"])
        if [row["index"] for row in data] != list(range(len(texts))):
            raise ValueError("Embedding count/order mismatch")
        vectors = [row["embedding"] for row in data]
        if any(
            len(vector) != DIMENSIONS
            or any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in vector)
            or not any(vector)
            for vector in vectors
        ):
            raise ValueError("Invalid embedding vector")
        # Embeddings do not generate output tokens.
        return vectors, replace(result, usage=result.usage.model_copy(update={"output_tokens": 0}))
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Embedding response invalid: %s", type(exc).__name__)
        raise RAGError("Mistral embeddings returned invalid dimensions or data; use mistral-embed (1024 dimensions)") from exc


def alias_target() -> str | None:
    try:
        aliases = qdrant("GET", "/aliases")["result"]["aliases"]
        target = next((row["collection_name"] for row in aliases if row["alias_name"] == settings().qdrant_alias), None)
        if target is not None and (not isinstance(target, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", target)):
            raise ValueError("Invalid collection name")
        return target
    except (KeyError, TypeError, ValueError) as exc:
        raise RAGError("Qdrant returned invalid alias metadata") from exc


def collection_prefix(data: Corpus) -> str:
    return f"{settings().qdrant_alias}-{data.version[:20]}-"


def index() -> dict:
    data = corpus()
    target = alias_target()
    if target and target.startswith(collection_prefix(data)):
        return {"status": "already_indexed", "documents": data.documents, "chunks": len(data.chunks)}
    # Each ingestion owns a new collection. Only the alias is replaced, after all points are durable.
    collection = collection_prefix(data) + uuid4().hex[:12]
    logger.info("Building Qdrant collection %s (%s chunks)", collection, len(data.chunks))
    qdrant("PUT", f"/collections/{collection}", {"vectors": {"size": DIMENSIONS, "distance": "Cosine"}})
    for offset in range(0, len(data.chunks), 32):
        batch = data.chunks[offset:offset + 32]
        vectors, response = embed([chunk["text"] for chunk in batch])
        logger.info("Embedded PDF batch: %s chunks, %s input tokens", len(batch), response.usage.input_tokens)
        qdrant("PUT", f"/collections/{collection}/points?wait=true", {"points": [
            {"id": chunk["id"], "vector": vector, "payload": chunk}
            for chunk, vector in zip(batch, vectors)
        ]})
    count = qdrant("POST", f"/collections/{collection}/points/count", {"exact": True})["result"].get("count")
    if count != len(data.chunks):
        raise RAGError("Qdrant index is incomplete; the active knowledge alias was not changed")
    actions = []
    if target:
        actions.append({"delete_alias": {"alias_name": settings().qdrant_alias}})
    actions.append({"create_alias": {"collection_name": collection, "alias_name": settings().qdrant_alias}})
    qdrant("POST", "/collections/aliases", {"actions": actions})
    return {"status": "indexed", "documents": data.documents, "chunks": len(data.chunks), "collection": collection}


def retrieve(question: str) -> tuple[list[dict], ai_providers.ProviderResult, dict]:
    started = perf_counter()
    with ai_tracing.span("rag_retrieval") as active:
        data = corpus()
        target = alias_target()
        if not target or not target.startswith(collection_prefix(data)):
            raise RAGError("PDF knowledge is not indexed or has changed. Run python -m backend.services.ai_rag index")
        with ai_tracing.span("rag_query_embedding") as embedding_span:
            vectors, response = embed([question])
            ai_tracing.record_usage(embedding_span, response)
        payload = qdrant("POST", f"/collections/{target}/points/query", {
            "query": vectors[0], "limit": 6, "score_threshold": SCORE_THRESHOLD,
            "with_payload": True, "with_vector": False,
        })
        try:
            points = payload["result"]["points"]
            if not isinstance(points, list) or len(points) > 6:
                raise ValueError("Invalid retrieval count")
            local = {chunk["id"]: chunk for chunk in data.chunks}
            sources = []
            seen = set()
            for point in points:
                chunk = local.get(str(point["id"]))
                score = point["score"]
                if chunk is None or point["payload"] != chunk or point["id"] in seen:
                    raise ValueError("Retrieved chunk does not match current local PDF corpus")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < SCORE_THRESHOLD:
                    raise ValueError("Invalid retrieval score")
                seen.add(point["id"])
                sources.append({**chunk, "source_id": f"S{len(sources) + 1}", "score": score})
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("RAG retrieval integrity failed: %s", type(exc).__name__)
            raise RAGError("Retrieved knowledge did not match the local PDFs; rebuild the index") from exc
        active.set_attributes({"retrieved_chunks": len(sources), "score_threshold": SCORE_THRESHOLD})
        return sources, response, {
            "corpus_version": data.version, "embedding_model": settings().embedding_model,
            "retrieved_chunks": len(sources), "score_threshold": SCORE_THRESHOLD,
            "latency_ms": (perf_counter() - started) * 1000,
        }


if __name__ == "__main__":
    if sys.argv[1:] != ["index"]:
        raise SystemExit("Usage: python -m backend.services.ai_rag index")
    logging.basicConfig(level=logging.INFO)
    try:
        print(json.dumps(index(), indent=2))
    except ai_providers.ProviderError as exc:
        logger.error("PDF indexing failed: %s", exc)
        raise SystemExit(1) from None
