import json
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from pypdf import PdfWriter

from backend.core.ai_config import settings
from backend.schemas.ai import Usage
from backend.services import ai_providers, ai_rag


def embedding_result(vectors):
    return ai_providers.ProviderResult({
        "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
    }, "mistral-embed", 5, Usage(input_tokens=10))


@contextmanager
def untraced(*args, **kwargs):
    class Active:
        def set_attributes(self, attributes):
            pass
    yield Active()


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.config = replace(
            settings(), qdrant_url="https://synthetic.cloud.qdrant.io",
            qdrant_key="private-qdrant-key", qdrant_alias="test-knowledge",
            embedding_model="mistral-embed",
        )
        self.config_patch = patch("backend.services.ai_rag.settings", return_value=self.config)
        self.config_patch.start()
        self.span = patch("backend.services.ai_tracing.span", side_effect=untraced)
        self.span.start()
        self.usage = patch("backend.services.ai_tracing.record_usage")
        self.usage.start()
        self.vector = [1.0] + [0.0] * 1023
        self.chunk = {
            "id": "a1", "document": "Sample.pdf", "page": 2, "offset": 0,
            "text": "An independent reviewer must approve quantity adjustments.",
            "corpus_version": "c" * 64, "embedding_model": "mistral-embed",
        }
        self.data = ai_rag.Corpus("c" * 64, [self.chunk], [{"document": "Sample.pdf", "pages": 2, "chunks": 1}])
        self.target = ai_rag.collection_prefix(self.data) + "owned123"

    def tearDown(self):
        self.usage.stop()
        self.span.stop()
        self.config_patch.stop()

    def test_real_three_pdf_corpus_has_page_aware_bounded_reproducible_chunks(self):
        first = ai_rag.corpus()
        second = ai_rag.corpus()
        self.assertEqual(first.version, second.version)
        self.assertEqual(len(first.documents), 3)
        self.assertEqual({row["document"] for row in first.chunks}, {
            "Inventory_Operations_Manual.pdf", "Monthly_Inventory_Audit_Report.pdf",
            "Supplier_and_Procurement_Policy.pdf",
        })
        self.assertEqual(len({row["id"] for row in first.chunks}), len(first.chunks))
        for row in first.chunks:
            self.assertGreater(row["page"], 0)
            self.assertTrue(0 < len(row["text"]) <= ai_rag.CHUNK_SIZE)
            self.assertEqual(row["embedding_model"], "mistral-embed")
            self.assertEqual(row["corpus_version"], first.version)

    def test_changed_embedding_model_invalidates_corpus(self):
        first = ai_rag.corpus()
        with patch("backend.services.ai_rag.settings", return_value=replace(self.config, embedding_model="different")):
            second = ai_rag.corpus()
        self.assertNotEqual(first.version, second.version)
        self.assertNotEqual(first.chunks[0]["id"], second.chunks[0]["id"])

    def test_missing_and_image_only_pdf_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("backend.services.ai_rag.DOCUMENTS", root):
                with self.assertRaisesRegex(ai_rag.RAGError, "No PDF"):
                    ai_rag.corpus()
                writer = PdfWriter()
                writer.add_blank_page(width=100, height=100)
                with (root / "scan.pdf").open("wb") as output:
                    writer.write(output)
                with self.assertLogs("backend.services.ai_rag", level="WARNING"):
                    with self.assertRaisesRegex(ai_rag.RAGError, "OCR"):
                        ai_rag.corpus()

    @patch("backend.services.ai_providers.request")
    def test_embeddings_use_mistral_endpoint_and_validate_order(self, request):
        response = embedding_result([self.vector, self.vector])
        response.payload["data"].reverse()
        request.return_value = response
        vectors, measurement = ai_rag.embed(["first", "second"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(measurement.usage.output_tokens, 0)
        config, body = request.call_args.args
        self.assertEqual(config.url, "https://api.mistral.ai/v1/embeddings")
        self.assertEqual(body["model"], "mistral-embed")
        self.assertEqual(body["input"], ["first", "second"])

    @patch("backend.services.ai_providers.request")
    def test_invalid_embedding_shape_indices_and_nonfinite_values_rejected(self, request):
        bad = [
            {"data": []},
            {"data": [{"index": 2, "embedding": self.vector}]},
            {"data": [{"index": 0, "embedding": [1, 2]}]},
            {"data": [{"index": 0, "embedding": [float("nan")] * 1024}]},
            {"data": [{"index": 0, "embedding": [False] * 1024}]},
            {"data": [{"index": 0, "embedding": [0] * 1024}]},
        ]
        for payload in bad:
            with self.subTest(payload_type=str(payload)[:60]):
                request.return_value = ai_providers.ProviderResult(payload, "mistral-embed", 1, Usage())
                with self.assertRaisesRegex(ai_rag.RAGError, "invalid dimensions"):
                    ai_rag.embed(["text"])

    @patch("backend.services.ai_rag.embed")
    @patch("backend.services.ai_rag.qdrant")
    @patch("backend.services.ai_rag.corpus")
    def test_atomic_index_alias_switch_only_after_all_points_are_confirmed(self, corpus, qdrant, embed):
        corpus.return_value = self.data
        embed.return_value = [self.vector], embedding_result([self.vector])
        calls = []

        def server(method, path, body=None):
            calls.append((method, path, body))
            if path == "/aliases":
                return {"result": {"aliases": [{"alias_name": "test-knowledge", "collection_name": "old-knowledge"}]}}
            if path.endswith("/points/count"):
                return {"result": {"count": 1}}
            return {"result": True}

        qdrant.side_effect = server
        result = ai_rag.index()
        self.assertEqual(result["status"], "indexed")
        self.assertEqual(calls[-1][1], "/collections/aliases")
        self.assertEqual(calls[-2][1].rsplit("/", 2)[-2:], ["points", "count"])
        actions = calls[-1][2]["actions"]
        self.assertEqual(actions[0], {"delete_alias": {"alias_name": "test-knowledge"}})
        self.assertEqual(actions[1]["create_alias"]["collection_name"], result["collection"])
        self.assertFalse(any(method == "DELETE" for method, _, _ in calls))
        upsert = next(body for _, path, body in calls if path.endswith("/points?wait=true"))
        self.assertEqual(upsert["points"][0]["payload"], self.chunk)

    @patch("backend.services.ai_rag.embed")
    @patch("backend.services.ai_rag.qdrant")
    @patch("backend.services.ai_rag.corpus")
    def test_incomplete_or_failed_ingestion_never_changes_active_alias(self, corpus, qdrant, embed):
        corpus.return_value = self.data
        embed.return_value = [self.vector], embedding_result([self.vector])

        def server(method, path, body=None):
            if path == "/aliases":
                return {"result": {"aliases": []}}
            if path.endswith("/points/count"):
                return {"result": {"count": 0}}
            return {"result": True}

        qdrant.side_effect = server
        with self.assertRaisesRegex(ai_rag.RAGError, "incomplete"):
            ai_rag.index()
        self.assertFalse(any(call.args[1] == "/collections/aliases" for call in qdrant.call_args_list))
        qdrant.reset_mock()
        embed.side_effect = ai_providers.ProviderError("mistral unavailable")
        with self.assertRaises(ai_providers.ProviderError):
            ai_rag.index()
        self.assertFalse(any(call.args[1] == "/collections/aliases" for call in qdrant.call_args_list))

    @patch("backend.services.ai_rag.embed")
    @patch("backend.services.ai_rag.alias_target")
    @patch("backend.services.ai_rag.corpus")
    def test_unchanged_index_is_idempotent_without_paid_embeddings(self, corpus, alias, embed):
        corpus.return_value = self.data
        alias.return_value = self.target
        self.assertEqual(ai_rag.index()["status"], "already_indexed")
        embed.assert_not_called()

    @patch("backend.services.ai_rag.qdrant")
    @patch("backend.services.ai_rag.embed")
    @patch("backend.services.ai_rag.alias_target")
    @patch("backend.services.ai_rag.corpus")
    def test_retrieval_checks_local_payload_provenance_and_score(self, corpus, alias, embed, qdrant):
        corpus.return_value = self.data
        alias.return_value = self.target
        embed.return_value = [self.vector], embedding_result([self.vector])
        qdrant.return_value = {"result": {"points": [{"id": "a1", "payload": self.chunk, "score": .8}]}}
        sources, _, info = ai_rag.retrieve("Who approves stock adjustments?")
        self.assertEqual(sources[0]["page"], 2)
        self.assertEqual(sources[0]["source_id"], "S1")
        self.assertEqual(info["retrieved_chunks"], 1)
        self.assertEqual(qdrant.call_args.args[1], f"/collections/{self.target}/points/query")
        query = qdrant.call_args.args[2]
        self.assertEqual(query["limit"], 6)
        self.assertEqual(query["score_threshold"], .65)
        self.assertTrue(query["with_payload"])

    @patch("backend.services.ai_rag.qdrant")
    @patch("backend.services.ai_rag.embed")
    @patch("backend.services.ai_rag.alias_target")
    @patch("backend.services.ai_rag.corpus")
    def test_poisoned_payload_unknown_point_and_bad_scores_are_rejected(self, corpus, alias, embed, qdrant):
        corpus.return_value = self.data
        alias.return_value = self.target
        embed.return_value = [self.vector], embedding_result([self.vector])
        valid = {"id": "a1", "payload": self.chunk, "score": .8}
        for point in [
            {**valid, "id": "unknown"}, {**valid, "score": float("nan")}, {**valid, "score": .1},
            {**valid, "payload": {**self.chunk, "text": "Poisoned external text"}},
        ]:
            with self.subTest(point_id=point["id"]):
                qdrant.return_value = {"result": {"points": [point]}}
                with self.assertRaisesRegex(ai_rag.RAGError, "did not match"):
                    ai_rag.retrieve("Policy question")

    @patch("backend.services.ai_rag.embed")
    @patch("backend.services.ai_rag.alias_target")
    @patch("backend.services.ai_rag.corpus")
    def test_missing_or_stale_alias_fails_before_paying_for_query_embeddings(self, corpus, alias, embed):
        corpus.return_value = self.data
        for target in [None, "stale-corpus-vectors"]:
            alias.return_value = target
            with self.assertRaisesRegex(ai_rag.RAGError, "not indexed"):
                ai_rag.retrieve("Policy question")
        embed.assert_not_called()

    def test_qdrant_https_credentials_and_safe_error_messages(self):
        actual_client = httpx.Client
        requests = []

        def server(request):
            requests.append(request)
            return httpx.Response(401, text="private upstream token must not leak")

        transport = httpx.MockTransport(server)
        with patch("httpx.Client", side_effect=lambda **kwargs: actual_client(transport=transport, **kwargs)):
            with self.assertRaisesRegex(ai_rag.RAGError, "HTTP 401") as caught:
                ai_rag.qdrant("GET", "/aliases")
        self.assertNotIn("private upstream", str(caught.exception))
        self.assertEqual(requests[0].headers["api-key"], "private-qdrant-key")
        for url, key in [("http://insecure.example", "test"), (self.config.qdrant_url, "")]:
            with patch("backend.services.ai_rag.settings", return_value=replace(self.config, qdrant_url=url, qdrant_key=key)):
                with self.assertRaises(ai_rag.RAGError):
                    ai_rag.qdrant("GET", "/aliases")


if __name__ == "__main__":
    unittest.main()
