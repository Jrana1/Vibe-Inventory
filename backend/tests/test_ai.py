import json
import os
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from starlette.requests import Request

from backend.core.ai_config import settings
from backend.core.database import Base, get_db
from backend.main import app
from backend.models import Inventory
from backend.routers.ai import gate, require_local
from backend.schemas.ai import (
    Evidence, PriceAction, PriceChange, PricePlan, QuantityAction, QuantityChange, QuantityPlan, RAGAnswer, RAGClaim,
    ROUTES, RetrievalQuestion, RouteDecision, SQLPlan, Usage,
)
from backend.schemas.inventory import QuantityUpdate
from backend.services import ai_agent, ai_comparison, ai_memory, ai_providers, ai_rag, ai_sql, ai_store, ai_tracing, inventory_service


def measured():
    return ai_providers.ProviderResult(
        {}, "synthetic-model", 12, Usage(input_tokens=100, output_tokens=20),
    )


def routed(agent="sql", confidence=.95):
    return RouteDecision(
        agent=agent, confidence=confidence,
        probabilities={name: float(name == agent) for name in ROUTES},
    ), measured()


def quantity_action(*pairs):
    return QuantityAction(updates=[
        QuantityChange(item_id=item_id, stock_quantity=quantity) for item_id, quantity in pairs
    ])


def retrieved():
    return ([{
        "source_id": "S1", "document": "Operations.pdf", "page": 2, "score": .85,
        "text": "An independent reviewer must approve each recorded stock adjustment.",
        "id": "synthetic-chunk",
    }], measured(), {
        "corpus_version": "synthetic-version", "embedding_model": "synthetic-embed",
        "retrieved_chunks": 1, "score_threshold": .65, "latency_ms": 2,
    })


class AITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="vibe-inventory-ai-")
        cls.path = Path(cls.directory.name) / "inventory.db"
        cls.env = patch.dict(os.environ, {
            "AI_ENABLED": "true", "TYPESAFE_API_KEY": "test-only", "MISTRAL_API_KEY": "test-only",
            "MLFLOW_TRACKING_URI": f"sqlite:///{(Path(cls.directory.name) / 'traces.db').as_posix()}",
            "MLFLOW_DISABLE_TELEMETRY": "true", "JEV_CONFIDENCE_THRESHOLD": ".8",
            "LANGSMITH_ENABLED": "false",
        })
        cls.env.start()
        settings.cache_clear()
        import mlflow

        cls.artifact_root = patch(
            "mlflow.tracking._tracking_service.utils.DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH",
            str(Path(cls.directory.name) / "default-artifacts"),
        )
        cls.artifact_root.start()
        mlflow.set_tracking_uri(settings().tracking_uri)
        with patch("backend.services.ai_tracing.ROOT", Path(cls.directory.name)):
            ai_tracing.initialize()

    @classmethod
    def tearDownClass(cls):
        from mlflow.tracking._tracking_service.utils import _get_store

        # Release this test store's pooled SQLite handles before Windows directory cleanup.
        _get_store(settings().tracking_uri).engine.dispose()
        cls.artifact_root.stop()
        cls.env.stop()
        settings.cache_clear()
        cls.directory.cleanup()

    def setUp(self):
        self.engine = create_engine(
            f"sqlite:///{self.path.as_posix()}", connect_args={"check_same_thread": False},
        )
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as database:
            database.add_all([
                Inventory(item_id=1, item_name="Demo drill", category="Tools", supplier="Alpha", price=10, stock_quantity=5),
                Inventory(item_id=2, item_name="Demo saw", category="Tools", supplier="Beta", price=20, stock_quantity=0),
                Inventory(item_id=3, item_name="Unpriced", category="Other", supplier="Alpha", price=None, stock_quantity=-1),
                Inventory(item_id=4, item_name="Tied price", category="Other", supplier="Beta", price=20, stock_quantity=50),
            ])
            database.commit()

        def override_database():
            with Session(self.engine) as database:
                yield database

        app.dependency_overrides[get_db] = override_database
        app.dependency_overrides[require_local] = lambda: None
        self.sql_path = patch("backend.services.ai_sql.DATABASE_PATH", self.path)
        self.sql_path.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.sql_path.stop()
        app.dependency_overrides.clear()
        self.engine.dispose()

    def quantities(self):
        with Session(self.engine) as database:
            return {item.item_id: item.stock_quantity for item in database.scalars(select(Inventory))}

    def test_project_name_and_database_path(self):
        from backend.core.database import DATABASE_PATH

        self.assertEqual(self.client.get("/openapi.json").json()["info"]["title"], "Vibe Inventory API")
        self.assertEqual(DATABASE_PATH.name, "vibe_inventory.db")

    def test_shadow_off_preserves_single_router(self):
        with patch.object(ai_providers, "route", return_value=routed("unsupported")) as jev, \
                patch.object(ai_providers, "route_mistral") as mistral:
            response = self.client.post("/api/ai/chat", json={"message": "Delete item 12"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["comparison_id"])
        jev.assert_called_once()
        mistral.assert_not_called()
        self.assertEqual(self.client.get("/api/ai/comparisons").json()["live"]["paired_samples"], 0)

    def test_shadow_uses_same_memory_context_once_and_never_dispatches_mistral(self):
        with patch.object(ai_providers, "route", return_value=routed("unsupported")) as jev, \
                patch.object(ai_providers, "route_mistral", return_value=routed("quantity")) as mistral, \
                patch.object(ai_agent, "dispatch") as dispatch:
            first = self.client.post("/api/ai/chat", json={"message": "Delete item 1"}).json()
            jev.reset_mock()
            response = self.client.post("/api/ai/chat", json={
                "message": "Set it to 10", "parent_run_id": first["id"], "compare_routing": True,
            })
        result = response.json()
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["memory"]["run_ids"], [first["id"]])
        jev.assert_called_once()
        mistral.assert_called_once_with(jev.call_args.args[0])
        self.assertIn("Set it to 10", jev.call_args.args[0])
        dispatch.assert_not_called()
        detail = self.client.get(f"/api/ai/comparisons/{result['comparison_id']}").json()
        self.assertEqual(detail["pairs"][0]["jev"]["agent"], "unsupported")
        self.assertEqual(detail["pairs"][0]["mistral"]["agent"], "quantity")
        self.assertEqual(detail["run_id"], result["id"])
        self.assertEqual(detail["metrics"]["agreement_rate"], 0)
        self.assertEqual([entry["task"] for entry in result["usage"] if "task" in entry], ["shadow_router"])
        self.assertEqual(self.quantities()[1], 5)

    def test_shadow_failure_does_not_invalidate_jev_dispatch(self):
        with patch.object(ai_providers, "route", return_value=routed("sql")), \
                patch.object(ai_providers, "route_mistral", side_effect=ai_providers.ProviderError("mistral timed out")), \
                patch.object(ai_agent, "dispatch") as dispatch, Session(self.engine) as database:
            events = list(ai_agent.chat_events(database, "Show stock", explain=False, compare_routing=True))
        dispatch.assert_called_once()
        self.assertEqual(events[-1]["data"]["status"], "completed")
        self.assertTrue(any(event["event"] == "warning" for event in events))
        detail = self.client.get(f"/api/ai/comparisons/{events[-1]['data']['comparison_id']}").json()
        self.assertEqual(detail["metrics"]["paired_samples"], 0)
        self.assertEqual(detail["metrics"]["providers"]["mistral"]["errors"], 1)

    def test_jev_failure_never_falls_back_to_shadow(self):
        with patch.object(ai_providers, "route", side_effect=ai_providers.ProviderError("jev timed out")), \
                patch.object(ai_providers, "route_mistral", return_value=routed("quantity")) as shadow, \
                patch.object(ai_agent, "dispatch") as dispatch:
            result = self.client.post("/api/ai/chat", json={
                "message": "Set item 1 to 10", "compare_routing": True,
            }).json()
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["agent"])
        dispatch.assert_not_called()
        shadow.assert_called_once()
        detail = self.client.get(f"/api/ai/comparisons/{result['comparison_id']}").json()
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["metrics"]["providers"]["jev"]["errors"], 1)
        self.assertEqual(self.quantities()[1], 5)

    def test_stream_forwards_shadow_flag_and_strict_flag_validation(self):
        with patch.object(ai_providers, "route", return_value=routed("unsupported")), \
                patch.object(ai_providers, "route_mistral", return_value=routed("unsupported")) as mistral, \
                patch.object(ai_providers, "explain", return_value=iter([])):
            response = self.client.post("/api/ai/chat/stream", json={"message": "Delete item 1", "compare_routing": True})
        self.assertEqual(response.status_code, 200)
        self.assertIn('"comparison_id":', response.text)
        mistral.assert_called_once()
        self.assertEqual(self.client.post("/api/ai/chat", json={
            "message": "Show stock", "compare_routing": "false",
        }).status_code, 422)

    def test_benchmark_fixed_twenty_pairs_forty_calls_and_no_inventory_actions(self):
        from collections import Counter

        self.assertEqual(Counter(label for label, _ in ai_comparison.DATASET), {label: 4 for label in ROUTES})
        calls = []

        def classify(provider, message):
            calls.append((provider, message))
            label = next(label for label, question in ai_comparison.DATASET if question == message)
            return routed(label)

        with patch.object(ai_providers, "route", side_effect=lambda message: classify("jev", message)), \
                patch.object(ai_providers, "route_mistral", side_effect=lambda message: classify("mistral", message)), \
                patch.object(ai_agent, "dispatch") as dispatch, \
                patch.object(ai_store, "create_proposal") as proposal, Session(self.engine) as database:
            result = ai_comparison.create(database, "benchmark")
            ai_comparison.benchmark(database, result)
        self.assertEqual(len(calls), 40)
        self.assertEqual(calls[0][0], "jev")
        self.assertEqual(calls[2][0], "mistral")
        for offset in range(0, 40, 2):
            self.assertEqual(calls[offset][1], calls[offset + 1][1])
        dispatch.assert_not_called()
        proposal.assert_not_called()
        self.assertEqual(self.quantities(), {1: 5, 2: 0, 3: -1, 4: 50})
        saved = self.client.get(f"/api/ai/comparisons/{result['id']}").json()
        self.assertEqual(saved["completed"], 20)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["metrics"]["paired_samples"], 20)
        self.assertEqual(saved["metrics"]["agreement_rate"], 1)

    def test_benchmark_endpoint_gate_conflict_and_failure_release(self):
        self.assertTrue(gate.acquire(blocking=False))
        try:
            self.assertEqual(self.client.post("/api/ai/comparisons/benchmark").status_code, 429)
        finally:
            gate.release()
        with patch("backend.routers.ai.Thread") as thread, \
                patch.object(ai_comparison, "pair", side_effect=RuntimeError("synthetic failure")):
            thread.return_value.start.side_effect = lambda: thread.call_args.kwargs["target"]()
            response = self.client.post("/api/ai/comparisons/benchmark")
        self.assertEqual(response.status_code, 202)
        # The response is a stable start snapshot even when a worker finishes immediately.
        self.assertEqual(response.json()["status"], "running")
        failed = self.client.get(f"/api/ai/comparisons/{response.json()['id']}").json()
        self.assertEqual(failed["status"], "failed")
        self.assertIsNotNone(failed["error"])
        self.assertTrue(gate.acquire(blocking=False))
        gate.release()

    def test_benchmark_retains_partial_pairs_and_recovers_without_paid_retries(self):
        with patch.object(ai_providers, "route", return_value=routed()), \
                patch.object(ai_providers, "route_mistral", side_effect=[
                    routed(), ai_providers.ProviderError("synthetic unavailable"),
                ]), Session(self.engine) as database:
            result = ai_comparison.create(database, "benchmark")
            result["pairs"].append(ai_comparison.pair("Show stock", 0))
            ai_comparison.save(database, result)
            result["pairs"].append(ai_comparison.pair("Show stock", 1))
            ai_comparison.save(database, result)
        with Session(self.engine) as database, patch.object(ai_providers, "route") as no_retry:
            ai_comparison.recover(database)
        no_retry.assert_not_called()
        detail = self.client.get(f"/api/ai/comparisons/{result['id']}").json()
        self.assertEqual(detail["status"], "interrupted")
        self.assertEqual(detail["completed"], 2)
        self.assertEqual(detail["metrics"]["paired_samples"], 1)
        self.assertEqual(detail["metrics"]["providers"]["mistral"]["errors"], 1)
        self.assertEqual(self.client.get("/api/ai/comparisons/not-found").status_code, 404)

    def test_benchmark_thread_start_failure_is_terminal_and_releases_gate(self):
        with patch("backend.routers.ai.Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("synthetic thread startup failure")
            with self.assertRaises(RuntimeError):
                self.client.post("/api/ai/comparisons/benchmark")
        saved = self.client.get("/api/ai/comparisons").json()["benchmarks"][0]
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["completed"], 0)
        self.assertIsNotNone(saved["error"])
        self.assertTrue(gate.acquire(blocking=False))
        gate.release()

    def test_mistral_router_identical_taxonomy_single_attempt_and_450_token_cap(self):
        response = measured()
        response.payload = {"choices": [{"message": {"content": routed()[0].model_dump_json()}}]}
        with patch.object(ai_providers, "request", return_value=response) as request:
            decision, _ = ai_providers.route_mistral("Show stock")
        config, body = request.call_args.args
        self.assertEqual(config.name, "mistral")
        self.assertEqual(body["messages"][1]["content"], "Show stock")
        self.assertIn("Classify", body["messages"][0]["content"])
        for label, definition in ROUTES.items():
            self.assertIn(label, body["messages"][0]["content"])
            self.assertIn(definition, body["messages"][0]["content"])
        self.assertEqual(body["max_tokens"], 450)
        self.assertEqual(request.call_args.kwargs["max_attempts"], 1)
        self.assertEqual(decision.agent, "sql")

    def proposal(self, action=None):
        with Session(self.engine) as database:
            result = ai_store.create_proposal(database, "run", action or quantity_action((1, 80), (2, 25)))
            database.commit()
            return result

    def test_mlflow_artifact_uri_is_local_and_initialization_is_idempotent(self):
        import mlflow

        experiment = mlflow.get_experiment_by_name("vibe-inventory-agent")
        self.assertEqual(
            experiment.artifact_location, (Path(self.directory.name) / "mlartifacts").as_uri(),
        )
        ai_tracing.initialize()
        self.assertEqual(
            mlflow.get_experiment_by_name("vibe-inventory-agent").experiment_id,
            experiment.experiment_id,
        )

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_sql_route_executes_maximum_with_ties_once(self, route, generate):
        route.return_value = routed()
        generate.return_value = SQLPlan(
            sql="SELECT item_name, price FROM inventory WHERE price = (SELECT MAX(price) FROM inventory) ORDER BY item_id",
        ), measured()
        result = self.client.post("/api/ai/chat", json={"message": "Show products with maximum unit price"}).json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["agent"], "sql")
        self.assertEqual(result["query"]["rows"], [["Demo saw", 20], ["Tied price", 20]])
        self.assertEqual(result["query"]["row_count"], 2)
        self.assertFalse(result["query"]["truncated"])
        generate.assert_called_once()
        self.assertIs(generate.call_args.args[2], SQLPlan)
        self.assertEqual(self.client.get(f"/api/ai/runs/{result['id']}").json()["id"], result["id"])
        import mlflow
        trace = mlflow.get_trace(result["trace_id"])
        self.assertIsNotNone(trace)
        serialized = trace.to_json()
        for sensitive in ("test-only", "Demo saw", "SELECT item_name"):
            self.assertNotIn(sensitive, serialized)

    def test_sql_distinct_aggregates_filters_and_nulls(self):
        self.assertEqual(ai_sql.execute("SELECT DISTINCT supplier FROM inventory ORDER BY supplier")["rows"], [["Alpha"], ["Beta"]])
        result = ai_sql.execute("SELECT supplier, SUM(stock_quantity) AS units FROM inventory GROUP BY supplier ORDER BY supplier")
        self.assertEqual(result["rows"], [["Alpha", 4], ["Beta", 50]])
        self.assertEqual(ai_sql.execute("SELECT price FROM inventory WHERE item_id = 3")["rows"], [[None]])
        self.assertEqual(ai_sql.execute("SELECT COUNT(*) AS count FROM inventory WHERE price > 1000")["rows"], [[0]])
        self.assertEqual(ai_sql.execute("SELECT item_id FROM inventory WHERE supplier LIKE 'Al%' ORDER BY item_id")["rows"], [[1], [3]])

    def test_sql_clamps_rows_and_reports_truncation(self):
        with Session(self.engine) as database:
            database.add_all([
                Inventory(item_id=number, item_name="Synthetic", category="Demo", supplier="Demo", price=1, stock_quantity=1)
                for number in range(10, 160)
            ])
            database.commit()
        result = ai_sql.execute("SELECT item_id FROM inventory ORDER BY item_id")
        self.assertEqual(result["row_count"], 100)
        self.assertEqual(len(result["rows"]), 100)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["rows"][0], [1])
        self.assertFalse(ai_sql.execute("SELECT item_id FROM inventory LIMIT 2")["truncated"])

    def test_sql_rejects_writes_exfiltration_and_unsupported_features(self):
        statements = [
            "DELETE FROM inventory", "UPDATE inventory SET stock_quantity=0",
            "INSERT INTO inventory(item_id) VALUES(99)", "DROP TABLE inventory",
            "PRAGMA table_info(inventory)", "ATTACH DATABASE 'other.db' AS other",
            "SELECT * FROM inventory; DELETE FROM inventory",
            "SELECT * FROM agent_runs", "SELECT * FROM sqlite_master",
            "SELECT * FROM main.inventory", "SELECT * FROM pragma_table_info('inventory')",
            "SELECT load_extension('anything') FROM inventory",
            "SELECT readfile('anything') FROM inventory",
            "SELECT randomblob(100000000) FROM inventory",
            "WITH RECURSIVE x AS (SELECT 1 UNION ALL SELECT 1 FROM x) SELECT * FROM x",
            "SELECT a.item_id FROM inventory a JOIN inventory b ON 1=1",
            "SELECT item_id FROM inventory UNION SELECT item_id FROM inventory",
            "SELECT (SELECT result FROM agent_runs LIMIT 1) FROM inventory",
            "SELECT CAST(item_name AS BLOB) FROM inventory",
            "SELECT 1e999 FROM inventory",
        ]
        for sql in statements:
            with self.subTest(sql=sql), self.assertRaises(ai_sql.SQLQueryError):
                ai_sql.execute(sql)
        self.assertEqual(self.quantities(), {1: 5, 2: 0, 3: -1, 4: 50})

    def test_sqlite_authorizer_independently_rejects_writes(self):
        # Bypass the parser deliberately to prove SQLite is also a separate boundary.
        with patch("backend.services.ai_sql.validate", return_value="SELECT result FROM agent_runs"):
            with self.assertRaises(ai_sql.SQLQueryError):
                ai_sql.execute("ignored")

    def test_sql_query_budget_interrupts_expensive_select(self):
        with Session(self.engine) as database:
            database.add_all([
                Inventory(item_id=n, item_name="Demo", category="Demo", supplier="Demo", price=1, stock_quantity=1)
                for n in range(10, 1010)
            ])
            database.commit()
        with self.assertRaises(ai_sql.SQLQueryError):
            ai_sql.execute(
                "SELECT SUM((SELECT SUM(i.stock_quantity) FROM inventory i WHERE i.item_id <= o.item_id)) FROM inventory o"
            )

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_invalid_generated_sql_is_explicit_failure(self, route, generate):
        route.return_value = routed()
        generate.return_value = SQLPlan(sql="DELETE FROM inventory"), measured()
        result = self.client.post("/api/ai/chat", json={"message": "Show inventory"}).json()
        self.assertEqual(result["status"], "failed")
        self.assertIn("SELECT", result["answer"])
        self.assertIsNone(result["query"])
        self.assertEqual(self.quantities()[1], 5)

    @patch("backend.services.ai_rag.retrieve", side_effect=lambda message: retrieved())
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_faq_agent_returns_grounded_pdf_answer_with_verified_quotes(self, route, generate, retrieve):
        route.return_value = routed("faq")
        generate.return_value = RAGAnswer(claims=[RAGClaim(
            text="Stock adjustments require independent review.",
            evidence=[Evidence(source_id="S1", quote_id="Q1")],
        )]), measured()
        with patch("backend.services.ai_sql.execute") as sql:
            result = self.client.post("/api/ai/chat", json={"message": "Who reviews stock adjustments?"}).json()
            sql.assert_not_called()
        self.assertEqual(result["agent"], "faq")
        self.assertIn("independent review", result["answer"])
        self.assertIn("[S1]", result["answer"])
        self.assertEqual(result["sources"][0]["document"], "Operations.pdf")
        self.assertEqual(result["sources"][0]["page"], 2)
        self.assertEqual(len(result["usage"]), 3)
        self.assertEqual(result["retrieval"]["retrieved_chunks"], 1)
        self.assertEqual(result["proposals"], [])
        self.assertIs(generate.call_args.args[2], RAGAnswer)
        self.assertEqual(
            ai_agent.explanation_facts(result)["document_sources"], result["sources"],
        )
        self.assertEqual(generate.call_args.kwargs["timeout"], 45)
        self.assertEqual(result["sources"][0]["quotes"], [retrieved()[0][0]["text"]])
        import mlflow

        serialized = mlflow.get_trace(result["trace_id"]).to_json()
        for sensitive in ("Who reviews stock adjustments?", "An independent reviewer", "Operations.pdf"):
            self.assertNotIn(sensitive, serialized)

    @patch("backend.services.ai_rag.retrieve", side_effect=lambda message: retrieved())
    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_rag_stream_preserves_citations_and_rejects_unknown_explanation_labels(self, route, generate, explain, retrieve):
        route.return_value = routed("faq")
        generate.return_value = RAGAnswer(claims=[RAGClaim(
            text="Adjustments require independent review.",
            evidence=[Evidence(source_id="S1", quote_id="Q1")],
        )]), measured()
        for label, expected in [("[S1]", "completed"), ("[S99]", "failed"), ("", "failed")]:
            with self.subTest(label=label):
                explain.return_value = iter(["Independent review is required. " + label, measured()])
                events = self.parse_events(self.client.post("/api/ai/chat/stream", json={"message": "Who reviews adjustments?"}))
                result = events[-1]["data"]
                self.assertEqual(result["explanation_status"], expected)
                self.assertEqual(len(result["usage"]), 4)
                self.assertIn("[S1]", result["answer"])
                self.assertEqual(result["sources"][0]["page"], 2)
                if expected == "failed":
                    self.assertEqual(result["explanation"], "")
                    self.assertIn("citations", result["explanation_error"])

    @patch("backend.services.ai_rag.retrieve", side_effect=lambda message: retrieved())
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_unknown_faq_is_not_invented(self, route, generate, retrieve):
        route.return_value = routed("faq")
        generate.return_value = RAGAnswer(), measured()
        result = self.client.post("/api/ai/chat", json={"message": "Do you ship to Mars?"}).json()
        self.assertIn("enough evidence", result["answer"])
        self.assertEqual(result["sources"], [])

    @patch("backend.services.ai_rag.retrieve", side_effect=lambda message: ([], measured(), {"retrieved_chunks": 0}))
    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_no_retrieved_evidence_skips_generation_and_streamed_explanation(self, route, generate, explain, retrieve):
        route.return_value = routed("faq")
        result = self.parse_events(self.client.post("/api/ai/chat/stream", json={"message": "Unknown policy?"}))[-1]["data"]
        self.assertIn("could not find relevant evidence", result["answer"])
        generate.assert_not_called()
        explain.assert_not_called()

    @patch("backend.services.ai_rag.retrieve", side_effect=lambda message: retrieved())
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_unknown_source_or_fabricated_excerpt_id_fails_explicitly(self, route, generate, retrieve):
        route.return_value = routed("faq")
        for source_id, quote_id in [
            ("S6", "Q1"), ("S1", "Q999"),
        ]:
            with self.subTest(source_id=source_id):
                generate.return_value = RAGAnswer(claims=[RAGClaim(
                    text="Unverified claim", evidence=[Evidence(source_id=source_id, quote_id=quote_id)],
                )]), measured()
                result = self.client.post("/api/ai/chat", json={"message": "Explain the adjustment policy"}).json()
                self.assertEqual(result["status"], "failed")
                self.assertIn("unsupported citations", result["answer"])
                self.assertEqual(result["sources"], [])

    @patch("backend.services.ai_rag.retrieve", side_effect=ai_rag.RAGError("PDF knowledge is not indexed"))
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_unconfigured_rag_is_an_explicit_saved_failure_without_dummy_fallback(self, route, generate, retrieve):
        route.return_value = routed("faq")
        result = self.client.post("/api/ai/chat", json={"message": "Explain procurement policy"}).json()
        self.assertEqual(result["status"], "failed")
        self.assertIn("not indexed", result["answer"])
        generate.assert_not_called()

    @patch("backend.services.ai_providers.generate")
    def test_paid_starter_generation_endpoint_is_removed(self, generate):
        self.assertEqual(self.client.post("/api/ai/suggestions", json={}).status_code, 404)
        generate.assert_not_called()

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_price_agent_previews_null_to_absolute_price_then_requires_approval(self, route, generate):
        route.return_value = routed("price")
        generate.return_value = PricePlan(updates=[PriceChange(item_id=3, price=120)]), measured()
        result = self.client.post("/api/ai/chat", json={
            "message": "can you set price of product id 3 fro null to 120?",
        }).json()
        self.assertEqual(result["status"], "completed", result["answer"])
        self.assertEqual(result["agent"], "price")
        proposal = result["proposals"][0]
        self.assertEqual(proposal["action"]["kind"], "price_batch")
        self.assertEqual(proposal["before"], [{"item_id": 3, "item_name": "Unpriced", "price": None}])
        self.assertEqual(proposal["after"][0]["price"], 120)
        with Session(self.engine) as database:
            self.assertIsNone(database.get(Inventory, 3).price)
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 200)
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 200)
        with Session(self.engine) as database:
            self.assertEqual(database.get(Inventory, 3).price, 120)
            self.assertEqual(database.get(Inventory, 3).stock_quantity, -1)

    def price_proposal(self, *pairs):
        with Session(self.engine) as database:
            result = ai_store.create_proposal(database, "price-run", PriceAction(updates=[
                PriceChange(item_id=item_id, price=price) for item_id, price in pairs
            ]))
            database.commit()
        return result

    def test_price_schema_rejects_invalid_values_duplicates_and_oversized_batches(self):
        for value in [-1, True, "120", None, float("nan"), float("inf"), 1.001, 1000001]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PriceChange(item_id=1, price=value)
        for value in [0, 0.01, 19.99, 120, 1000000]:
            self.assertEqual(PriceChange(item_id=1, price=value).price, value)
        for updates in [
            [], [PriceChange(item_id=1, price=10)] * 2,
            [PriceChange(item_id=index, price=10) for index in range(1, 27)],
        ]:
            with self.assertRaises(ValidationError):
                PriceAction(updates=updates)
        with self.assertRaises(ValidationError):
            PricePlan(updates=[PriceChange(item_id=1, price=10)], clarification="Missing value")
        with self.assertRaises(ValidationError):
            PricePlan()

    def test_price_batch_is_atomic_and_does_not_change_quantities(self):
        proposal = self.price_proposal((1, 19.99), (3, 0))
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 200)
        with Session(self.engine) as database:
            self.assertEqual(database.get(Inventory, 1).price, 19.99)
            self.assertEqual(database.get(Inventory, 3).price, 0)
        self.assertEqual(self.quantities(), {1: 5, 2: 0, 3: -1, 4: 50})

    def test_stale_null_price_prevents_entire_price_batch(self):
        proposal = self.price_proposal((1, 30), (3, 120))
        with Session(self.engine) as database:
            item = database.get(Inventory, 3)
            item.price = 99
            database.commit()
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 409)
        with Session(self.engine) as database:
            self.assertEqual(database.get(Inventory, 1).price, 10)
            self.assertEqual(database.get(Inventory, 3).price, 99)
            self.assertEqual(database.get(ai_store.AgentProposal, proposal["id"]).status, "pending")

    def test_price_batch_execution_failure_rolls_back_all_items(self):
        proposal = self.price_proposal((1, 30), (3, 120))
        original = inventory_service.update_price

        def fail_second(database, item_id, update, **kwargs):
            if item_id == 3:
                raise HTTPException(400, "Injected second price failure")
            return original(database, item_id, update, **kwargs)

        with patch("backend.services.inventory_service.update_price", side_effect=fail_second):
            self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 400)
        with Session(self.engine) as database:
            self.assertEqual(database.get(Inventory, 1).price, 10)
            self.assertIsNone(database.get(Inventory, 3).price)
            self.assertEqual(database.get(ai_store.AgentProposal, proposal["id"]).status, "pending")

    def test_price_rejection_expiration_and_missing_items(self):
        proposal = self.price_proposal((3, 120))
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/reject").status_code, 200)
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 409)
        expired = self.price_proposal((3, 120))
        with Session(self.engine) as database:
            database.get(ai_store.AgentProposal, expired["id"]).expires_at = time.time() - 1
            database.commit()
        self.assertEqual(self.client.post(f"/api/ai/proposals/{expired['id']}/approve").status_code, 409)
        with self.assertRaises(HTTPException):
            self.price_proposal((1, 30), (999, 120))
        with Session(self.engine) as database:
            self.assertEqual(database.get(Inventory, 1).price, 10)
            self.assertIsNone(database.get(Inventory, 3).price)

    def test_price_pending_proposals_survive_startup_and_concurrent_approval_is_idempotent(self):
        proposal = self.price_proposal((3, 120))
        with Session(self.engine) as database:
            ai_store.recover_interrupted_runs(database)
            self.assertEqual(database.get(ai_store.AgentProposal, proposal["id"]).status, "pending")

        def approve(_):
            with Session(self.engine) as database:
                return ai_store.decide(database, proposal["id"], True, "price-test-trace")["status"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(list(executor.map(approve, range(2))), ["approved", "approved"])
        with Session(self.engine) as database:
            self.assertEqual(database.get(Inventory, 3).price, 120)

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_price_relative_changes_and_ambiguous_targets_clarify_without_generation(self, route, generate):
        parent = self.memory_run("SELECT item_id FROM inventory WHERE item_id IN (1, 2)")
        route.return_value = routed("price")
        for message in [
            "Set this product price to 120", "Increase price of item 1 by 20",
            "Increase price of item 1 by 10%", "Set item 1 price to 110 percent",
        ]:
            result = self.client.post("/api/ai/chat", json={
                "message": message, "parent_run_id": parent,
            }).json()
            self.assertEqual(result["status"], "needs_clarification")
            self.assertEqual(result["proposals"], [])
        generate.assert_not_called()

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_price_session_group_requires_every_target_and_current_ids_override(self, route, generate):
        parent = self.memory_run("SELECT item_id FROM inventory WHERE item_id IN (1, 2)")
        route.return_value = routed("price")
        for pairs, expected in [
            ([(1, 120), (2, 120)], "completed"), ([(1, 120)], "needs_clarification"),
            ([(1, 120), (2, 120), (3, 120)], "needs_clarification"),
        ]:
            generate.return_value = PricePlan(updates=[
                PriceChange(item_id=item_id, price=price) for item_id, price in pairs
            ]), measured()
            result = self.client.post("/api/ai/chat", json={
                "message": "Set the price of both products to 120", "parent_run_id": parent,
            }).json()
            self.assertEqual(result["status"], expected, result["answer"])
        generate.return_value = PricePlan(updates=[PriceChange(item_id=3, price=120)]), measured()
        result = self.client.post("/api/ai/chat", json={
            "message": "Set price of item 3 to 120", "parent_run_id": parent,
        }).json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["proposals"][0]["after"][0]["item_id"], 3)

    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_price_stream_explanation_failure_preserves_preview_and_memory(self, route, generate, explain):
        parent = self.memory_run("SELECT item_id FROM inventory WHERE item_id = 3")
        route.return_value = routed("price")
        generate.return_value = PricePlan(updates=[PriceChange(item_id=3, price=120)]), measured()

        def interrupted(*args):
            yield "Review the unpriced item."
            raise ai_providers.ProviderError("Explanation unavailable")

        explain.side_effect = interrupted
        events = self.parse_events(self.client.post("/api/ai/chat/stream", json={
            "message": "Set its price to 120", "parent_run_id": parent,
        }))
        result = events[-1]["data"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["explanation_status"], "failed")
        self.assertEqual(result["proposals"][0]["status"], "pending")
        facts = explain.call_args.args[1]
        self.assertEqual(facts["preview"][0]["kind"], "price_batch")
        self.assertIsNone(facts["preview"][0]["before"][0]["price"])
        with Session(self.engine) as database:
            self.assertIsNone(database.get(Inventory, 3).price)
            memory = ai_memory.load(database, result["id"])
            self.assertEqual(memory.item_ids, [3])
        saved = self.client.get(f"/api/ai/runs/{result['id']}").json()
        self.assertEqual(saved["agent"], "price")
        self.assertIn("price_agent", [stage["name"] for stage in saved["stages"]])

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_price_plan_clarification_does_not_create_a_partial_preview(self, route, generate):
        route.return_value = routed("price")
        generate.return_value = PricePlan(clarification="Please provide the final unit price."), measured()
        result = self.client.post("/api/ai/chat", json={"message": "Set item 3 price"}).json()
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["proposals"], [])

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_quantity_agent_previews_then_approves_entire_batch(self, route, generate):
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 80), (2, 25)).updates), measured()
        result = self.client.post("/api/ai/chat", json={"message": "Set item 1 to 80 and item 2 to 25"}).json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["agent"], "quantity")
        self.assertEqual(self.quantities()[1], 5)
        proposal = result["proposals"][0]
        self.assertEqual([row["stock_quantity"] for row in proposal["after"]], [80, 25])
        self.assertNotIn("price", proposal["before"][0])
        response = self.client.post(f"/api/ai/proposals/{proposal['id']}/approve")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.quantities(), {1: 80, 2: 25, 3: -1, 4: 50})
        updated = self.client.get(f"/api/ai/runs/{result['id']}").json()
        self.assertEqual(updated["proposals"][0]["status"], "approved")

    def test_single_quantity_change_and_repeat_approval(self):
        proposal = self.proposal(quantity_action((1, 80)))
        endpoint = f"/api/ai/proposals/{proposal['id']}/approve"
        self.assertEqual(self.client.post(endpoint).status_code, 200)
        self.assertEqual(self.client.post(endpoint).status_code, 200)
        self.assertEqual(self.quantities()[1], 80)
        self.assertEqual(self.quantities()[2], 0)

    def test_missing_item_prevents_whole_proposal(self):
        with Session(self.engine) as database:
            with self.assertRaises(HTTPException):
                ai_store.create_proposal(database, "run", quantity_action((1, 80), (999, 25)))
            database.rollback()
            self.assertEqual(database.scalars(select(ai_store.AgentProposal)).all(), [])
        self.assertEqual(self.quantities()[1], 5)

    def test_stale_item_prevents_all_batch_updates(self):
        proposal = self.proposal()
        with Session(self.engine) as database:
            inventory_service.update_quantity(database, 2, QuantityUpdate(stock_quantity=7))
        response = self.client.post(f"/api/ai/proposals/{proposal['id']}/approve")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.quantities()[1], 5)
        self.assertEqual(self.quantities()[2], 7)

    def test_execution_failure_rolls_back_earlier_batch_updates(self):
        proposal = self.proposal()
        original = inventory_service.update_quantity

        def fail_second(database, item_id, update, **kwargs):
            if item_id == 2:
                raise HTTPException(400, "Injected second-item failure")
            return original(database, item_id, update, **kwargs)

        with patch("backend.services.inventory_service.update_quantity", side_effect=fail_second):
            self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 400)
        self.assertEqual(self.quantities()[1], 5)
        with Session(self.engine) as database:
            self.assertEqual(database.get(ai_store.AgentProposal, proposal["id"]).status, "pending")

    def test_concurrent_approvals_are_idempotent(self):
        proposal = self.proposal()

        def approve(_):
            with Session(self.engine) as database:
                return ai_store.decide(database, proposal["id"], True, "test-trace")["status"]

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(list(executor.map(approve, range(2))), ["approved", "approved"])
        self.assertEqual(self.quantities()[1], 80)
        self.assertEqual(self.quantities()[2], 25)

    def test_rejection_expiration_and_missing_proposals(self):
        proposal = self.proposal()
        with Session(self.engine) as database:
            database.get(ai_store.AgentProposal, proposal["id"]).expires_at = time.time() - 1
            database.commit()
        endpoint = f"/api/ai/proposals/{proposal['id']}"
        self.assertEqual(self.client.post(endpoint + "/approve").status_code, 409)
        self.assertEqual(self.client.post(endpoint + "/reject").status_code, 200)
        self.assertEqual(self.client.post(endpoint + "/approve").status_code, 409)
        self.assertEqual(self.client.post("/api/ai/proposals/missing/approve").status_code, 404)
        self.assertEqual(self.quantities()[1], 5)

    def test_quantity_schema_rejects_invalid_or_partial_values(self):
        for updates in [
            [(1, -1)], [(1, 1.5)], [(True, 10)], [(1, 1000001)], [(1, 2), (1, 3)],
            [(n, 1) for n in range(1, 27)],
        ]:
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                quantity_action(*updates)
        with self.assertRaises(ValidationError):
            QuantityPlan(updates=quantity_action((1, 5)).updates, clarification="Missing item")

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_quantity_clarification_does_not_create_partial_proposal(self, route, generate):
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(clarification="Provide item IDs and final quantities."), measured()
        result = self.client.post("/api/ai/chat", json={"message": "Increase drill quantity"}).json()
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["proposals"], [])

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_session_followup_uses_verified_item_and_absolute_quantity(self, route, generate):
        route.return_value = routed("sql")
        generate.return_value = SQLPlan(
            sql="SELECT item_id, item_name, stock_quantity FROM inventory WHERE item_id = 1",
        ), measured()
        previous = self.client.post("/api/ai/chat", json={"message": "Show item 1"}).json()
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 10)).updates), measured()
        response = self.client.post("/api/ai/chat", json={
            "message": "Can you increase quantity of this product to 10?",
            "parent_run_id": previous["id"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["parent_run_id"], previous["id"])
        self.assertEqual(result["memory"]["item_ids"], [1])
        self.assertEqual(result["memory"]["run_ids"], [previous["id"]])
        self.assertEqual(result["proposals"][0]["after"][0]["stock_quantity"], 10)
        self.assertEqual(self.quantities()[1], 5)
        context = json.loads(route.call_args.args[0])
        self.assertEqual(context["current_request"], response.json()["request"])
        self.assertEqual(context["item_ids"], [1])
        self.assertEqual(context["recent_turns"][0]["request"], "Show item 1")
        self.assertIn("increase to", generate.call_args.args[0])
        self.assertIn("increase by", generate.call_args.args[0])

    def memory_run(self, sql="SELECT item_id, item_name FROM inventory WHERE item_id = 1", parent=None, **overrides):
        run_id = str(uuid4())
        result = {
            "schema_version": 2, "id": run_id, "request": "Show item 1",
            "status": "completed", "agent": "sql", "answer": "Returned inventory rows.",
            "query": ai_sql.execute(sql) if sql else None, "parent_run_id": parent,
            **overrides,
        }
        with Session(self.engine) as database:
            database.add(ai_store.AgentRun(
                id=run_id, created_at=time.time(), status=result["status"], result=json.dumps(result),
            ))
            database.commit()
        return run_id

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_session_followup_updates_both_verified_products_to_absolute_quantity(self, route, generate):
        route.return_value = routed("sql")
        generate.return_value = SQLPlan(
            sql="SELECT item_name, item_id FROM inventory WHERE item_id IN (1, 2) ORDER BY item_id",
        ), measured()
        previous = self.client.post("/api/ai/chat", json={
            "message": "Give me product IDs for Demo drill and Demo saw",
        }).json()
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 3), (2, 3)).updates), measured()
        result = self.client.post("/api/ai/chat", json={
            "message": "can you increase quantity for both this product to 3",
            "parent_run_id": previous["id"],
        }).json()
        self.assertEqual(result["memory"]["item_ids"], [1, 2])
        self.assertEqual(result["status"], "completed", result["answer"])
        proposal = result["proposals"][0]
        self.assertEqual(
            [(row["item_id"], row["stock_quantity"]) for row in proposal["after"]],
            [(1, 3), (2, 3)],
        )
        self.assertEqual(self.quantities(), {1: 5, 2: 0, 3: -1, 4: 50})
        self.assertEqual(json.loads(generate.call_args.args[1])["item_ids"], [1, 2])
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 200)
        self.assertEqual(self.quantities(), {1: 3, 2: 3, 3: -1, 4: 50})

    def test_session_selection_references_require_the_right_cardinality(self):
        cases = [
            ("Set this product to 3", [1], [1]),
            ("Set this product to 3", [1, 2], []),
            ("Set both this product to 3", [1, 2], [1, 2]),
            ("Set both of them to 3", [1], []),
            ("Set both of them to 3", [1, 2, 4], []),
            ("Set both products to 3", [], []),
            ("Set them to 3", [1, 2, 4], [1, 2, 4]),
            ("Set their quantities to 3", [1, 2], [1, 2]),
            ("Set these products to 3", [1, 2], [1, 2]),
            ("Set those items to 3", [1, 2], [1, 2]),
            ("Set the previous products to 3", [1, 2], [1, 2]),
            ("Set all of these products to 3", [1, 2], [1, 2]),
            ("Set all products to 3", [1, 2], []),
            ("Set some products to 3", [1, 2], []),
            ("Set these products to 3", [1], []),
            ("Set them to 3", list(range(1, 27)), []),
        ]
        for message, selection, expected in cases:
            with self.subTest(message=message, selection=selection):
                self.assertEqual(ai_memory.referenced_item_ids(message, selection), expected)

    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_plural_stream_followup_uses_the_full_group_and_rejects_partial_or_expanded_plans(
        self, route, generate, explain,
    ):
        parent = self.memory_run(
            "SELECT item_id, item_name FROM inventory WHERE item_id IN (1, 2, 4) ORDER BY item_id",
        )
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 3), (2, 3), (4, 3)).updates), measured()
        explain.return_value = iter(["Review all three items before approving.", measured()])
        events = self.parse_events(self.client.post("/api/ai/chat/stream", json={
            "message": "Increase quantity for these products to 3", "parent_run_id": parent,
        }))
        result = events[-1]["data"]
        self.assertEqual(result["status"], "completed", result["answer"])
        self.assertEqual(result["memory"]["item_ids"], [1, 2, 4])
        self.assertEqual(
            [(row["item_id"], row["stock_quantity"]) for row in result["proposals"][0]["after"]],
            [(1, 3), (2, 3), (4, 3)],
        )
        for pairs in [[(1, 3)], [(1, 3), (2, 3), (3, 3), (4, 3)]]:
            generate.return_value = QuantityPlan(updates=quantity_action(*pairs).updates), measured()
            blocked = self.client.post("/api/ai/chat", json={
                "message": "Set them to 3", "parent_run_id": parent,
            }).json()
            self.assertEqual(blocked["status"], "needs_clarification")
            self.assertEqual(blocked["proposals"], [])
        self.assertEqual(self.quantities(), {1: 5, 2: 0, 3: -1, 4: 50})

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_both_does_not_guess_a_pair_from_three_or_from_an_incomplete_result(self, route, generate):
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 3), (2, 3)).updates), measured()
        truncated = ai_sql.execute("SELECT item_id FROM inventory WHERE item_id IN (1, 2)")
        truncated["truncated"] = True
        parents = [
            self.memory_run("SELECT item_id FROM inventory WHERE item_id IN (1, 2, 4)"),
            self.memory_run(query=truncated),
        ]
        for parent in parents:
            result = self.client.post("/api/ai/chat", json={
                "message": "Set both products to 3", "parent_run_id": parent,
            }).json()
            self.assertEqual(result["status"], "needs_clarification")
            self.assertEqual(result["proposals"], [])
        generate.assert_not_called()

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_memory_ambiguous_empty_and_computed_product_ids_cannot_change_quantity(self, route, generate):
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 10)).updates), measured()
        for sql in [
            "SELECT item_id FROM inventory WHERE item_id IN (1, 2)",
            "SELECT item_id FROM inventory WHERE item_id = 999",
            "SELECT COUNT(*) AS item_id FROM inventory",
            "SELECT 1 AS item_id FROM inventory WHERE item_id = 2",
            "SELECT MIN(item_id) AS item_id FROM inventory",
            "SELECT item_id FROM (SELECT MIN(item_id) AS item_id FROM inventory)",
            "SELECT item_id, MAX(price) FROM inventory",
            "SELECT 1 AS item_id, inventory.* FROM inventory WHERE item_id = 2",
        ]:
            with self.subTest(sql=sql):
                parent = self.memory_run(sql)
                result = self.client.post("/api/ai/chat", json={
                    "message": "Set this product to 10", "parent_run_id": parent,
                }).json()
                self.assertEqual(result["status"], "needs_clarification")
                self.assertEqual(result["proposals"], [])
        generate.assert_not_called()
        self.assertEqual(self.quantities()[1], 5)

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_fresh_chat_does_not_use_another_conversations_product(self, route, generate):
        parent = self.memory_run()
        route.return_value = routed("quantity")
        for request in [{"message": "Set this product to 10"}, {"message": "Set it to 10", "parent_run_id": None}]:
            result = self.client.post("/api/ai/chat", json=request).json()
            self.assertEqual(result["status"], "needs_clarification")
            self.assertEqual(result["memory"], {"run_ids": [], "item_ids": []})
        generate.assert_not_called()
        with Session(self.engine) as database:
            self.assertEqual(ai_memory.load(database, parent).item_ids, [1])

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_current_explicit_id_overrides_memory_and_mismatched_plan_is_blocked(self, route, generate):
        parent = self.memory_run()
        route.return_value = routed("quantity")
        for item_id, expected in [(2, "completed"), (1, "needs_clarification")]:
            generate.return_value = QuantityPlan(updates=quantity_action((item_id, 10)).updates), measured()
            result = self.client.post("/api/ai/chat", json={
                "message": "Set item 2 to 10", "parent_run_id": parent,
            }).json()
            self.assertEqual(result["status"], expected)
            if result["proposals"]:
                self.assertEqual(result["proposals"][0]["before"][0]["item_id"], 2)
        self.assertEqual(self.quantities()[2], 0)

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_memory_does_not_enable_relative_changes(self, route, generate):
        parent = self.memory_run()
        route.return_value = routed("quantity")
        result = self.client.post("/api/ai/chat", json={
            "message": "Increase this product by 10", "parent_run_id": parent,
        }).json()
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["proposals"], [])
        self.assertIn("absolute", result["answer"])
        generate.assert_not_called()

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_memory_does_not_bind_unrelated_names_or_plural_targets_to_one_item(self, route, generate):
        parent = self.memory_run()
        route.return_value = routed("quantity")
        for message in ["Set all products to 10", "Set their quantities to 10", "Set Demo saw to 10"]:
            result = self.client.post("/api/ai/chat", json={
                "message": message, "parent_run_id": parent,
            }).json()
            self.assertEqual(result["status"], "needs_clarification")
            self.assertEqual(result["proposals"], [])
        generate.assert_not_called()

    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_streaming_followup_reloads_current_stock_and_approval_rechecks_it(self, route, generate, explain):
        parent = self.memory_run()
        with Session(self.engine) as database:
            inventory_service.update_quantity(database, 1, QuantityUpdate(stock_quantity=7))
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 10)).updates), measured()
        explain.return_value = iter(["Preview only.", measured()])
        events = self.parse_events(self.client.post("/api/ai/chat/stream", json={
            "message": "Increase this product to 10", "parent_run_id": parent,
        }))
        result = events[-1]["data"]
        self.assertEqual(result["parent_run_id"], parent)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["before"][0]["stock_quantity"], 7)
        self.assertEqual(proposal["after"][0]["stock_quantity"], 10)
        with Session(self.engine) as database:
            inventory_service.update_quantity(database, 1, QuantityUpdate(stock_quantity=8))
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 409)
        self.assertEqual(self.quantities()[1], 8)
        with Session(self.engine) as database:
            memory = ai_memory.load(database, result["id"])
        self.assertEqual(memory.item_ids, [1])
        self.assertEqual(memory.turns[-1]["decisions"][0]["status"], "pending")
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/reject").status_code, 200)
        with Session(self.engine) as database:
            memory = ai_memory.load(database, result["id"])
        self.assertEqual(memory.turns[-1]["decisions"][0]["status"], "rejected")

    def test_memory_is_bounded_and_does_not_restore_an_older_product_after_new_sql(self):
        parent = self.memory_run()
        latest = parent
        for index in range(7):
            latest = self.memory_run(
                sql=None, parent=latest, agent="faq", request=f"Policy question {index}",
                answer="A" * 1200, sources=[{"text": "Do not replay this quotation"}],
                explanation="Do not trust this streamed answer",
            )
        with Session(self.engine) as database:
            memory = ai_memory.load(database, latest)
        self.assertEqual(len(memory.turns), 6)
        self.assertEqual(memory.item_ids, [])
        self.assertNotIn(parent, memory.metadata()["run_ids"])
        self.assertNotIn("Do not replay", memory.prompt("Follow-up"))
        self.assertNotIn("Do not trust", memory.prompt("Follow-up"))
        self.assertTrue(all(turn["answer_incomplete"] for turn in memory.turns))
        recent = self.memory_run("SELECT COUNT(*) FROM inventory", parent=parent)
        with Session(self.engine) as database:
            self.assertEqual(ai_memory.load(database, recent).item_ids, [])
        latest = None
        for _ in range(6):
            latest = self.memory_run(sql=None, parent=latest, agent="faq", request="Q" * 2000, answer="A" * 1000)
        with Session(self.engine) as database:
            memory = ai_memory.load(database, latest)
        self.assertLess(len(memory.turns), 6)
        self.assertLessEqual(len(json.dumps(memory.turns)), ai_memory.MAX_HISTORY_CHARS)

    def test_memory_identity_requires_direct_ids_and_complete_results(self):
        for sql in [
            "SELECT item_id AS product_id FROM inventory WHERE item_id = 1",
            "SELECT inventory.* FROM inventory WHERE item_id = 1",
            "SELECT * FROM inventory WHERE item_id = 1",
            "SELECT item_id, price FROM inventory WHERE price = (SELECT MAX(price) FROM inventory)",
        ]:
            with self.subTest(sql=sql):
                data = ai_sql.execute(sql)
                expected = [row[0] for row in data["rows"]]
                self.assertEqual(ai_sql.result_item_ids(data), sorted(expected))
                self.assertEqual(ai_sql.result_item_ids({**data, "truncated": True}), [])

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_missing_invalid_or_running_parent_is_an_explicit_failure(self, route, generate):
        self.assertEqual(self.client.post("/api/ai/chat", json={
            "message": "Follow-up", "parent_run_id": "not-a-uuid",
        }).status_code, 422)
        running = self.memory_run(status="running")
        for parent in [str(uuid4()), running]:
            result = self.client.post("/api/ai/chat", json={
                "message": "Follow-up", "parent_run_id": parent,
            }).json()
            self.assertEqual(result["status"], "failed")
            self.assertIn("new chat", result["answer"])
        route.assert_not_called()
        generate.assert_not_called()

    @patch("backend.services.ai_rag.retrieve", return_value=retrieved())
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_faq_followup_rewrites_search_and_still_requires_fresh_citations(self, route, generate, retrieve):
        parent = self.memory_run(
            sql=None, agent="faq", request="What are the rules for recorded stock adjustments?",
            answer="Old answer is not new evidence.",
        )
        route.return_value = routed("faq")
        generate.side_effect = [
            (RetrievalQuestion(question="Who approves recorded stock adjustments?"), measured()),
            (RAGAnswer(claims=[RAGClaim(
                text="An independent reviewer must approve adjustments.",
                evidence=[Evidence(source_id="S1", quote_id="Q1")],
            )]), measured()),
        ]
        result = self.client.post("/api/ai/chat", json={
            "message": "Who approves them?", "parent_run_id": parent,
        }).json()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["sources"][0]["document"], "Operations.pdf")
        retrieve.assert_called_once_with("Who approves recorded stock adjustments?")
        self.assertIs(generate.call_args_list[0].args[2], RetrievalQuestion)
        self.assertNotIn("Old answer", generate.call_args_list[1].args[1])

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_sql_followup_receives_context_but_rereads_inventory(self, route, generate):
        parent = self.memory_run()
        route.return_value = routed("sql")
        generate.return_value = SQLPlan(sql="SELECT item_id, supplier FROM inventory WHERE item_id = 1"), measured()
        result = self.client.post("/api/ai/chat", json={
            "message": "Who supplies it?", "parent_run_id": parent,
        }).json()
        self.assertEqual(result["query"]["rows"], [[1, "Alpha"]])
        context = json.loads(generate.call_args.args[1])
        self.assertEqual(context["current_request"], "Who supplies it?")
        self.assertEqual(context["recent_turns"][0]["result"]["rows"], [[1, "Demo drill"]])
        self.assertEqual(context["item_ids"], [1])

    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_low_confidence_and_unsupported_routes_skip_agents(self, route, generate):
        for decision in [routed("sql", .2), routed("unsupported")]:
            route.return_value = decision
            result = self.client.post("/api/ai/chat", json={"message": "Do something"}).json()
            self.assertEqual(result["status"], "needs_clarification")
        generate.assert_not_called()

    @patch("backend.services.ai_providers.route", side_effect=ai_providers.ProviderError("jev key not configured"))
    def test_provider_failure_persisted_without_fallback(self, route):
        result = self.client.post("/api/ai/chat", json={"message": "List inventory"}).json()
        self.assertEqual(result["status"], "failed")
        self.assertIn("jev key", result["answer"])
        self.assertEqual(self.client.get("/api/ai/runs").json()[0]["status"], "failed")

    def test_startup_recovery_rejects_legacy_price_proposals_without_deleting_history(self):
        proposal = self.proposal()
        with Session(self.engine) as database:
            database.get(ai_store.AgentProposal, proposal["id"]).action = '{"kind":"price","item_ids":[1],"adjustment":5}'
            database.add(ai_store.AgentRun(id="old-run", created_at=0, status="completed", result='{"kind":"benchmark"}'))
            database.add(ai_store.AgentRun(id="interrupted", created_at=0, status="running", result='{"schema_version":2}'))
            database.commit()
            ai_store.recover_interrupted_runs(database)
            self.assertEqual(database.get(ai_store.AgentProposal, proposal["id"]).status, "rejected")
            self.assertIsNotNone(database.get(ai_store.AgentRun, "old-run"))
            self.assertEqual(database.get(ai_store.AgentRun, "interrupted").status, "failed")
        self.assertEqual(self.client.get("/api/ai/runs/old-run").status_code, 410)
        self.assertEqual(self.client.post(f"/api/ai/proposals/{proposal['id']}/approve").status_code, 409)

    def test_startup_recovery_marks_interrupted_explanation_without_invalidating_preview(self):
        proposal = self.proposal()
        with Session(self.engine) as database:
            database.add(ai_store.AgentRun(id="run", created_at=1, status="completed", result=json.dumps({
                "schema_version": 2, "explanation_status": "streaming", "explanation": "",
                "status": "completed", "proposals": [proposal],
            })))
            database.commit()
            ai_store.recover_interrupted_runs(database)
            record = database.get(ai_store.AgentRun, "run")
            self.assertEqual(record.status, "completed")
            self.assertEqual(json.loads(record.result)["explanation_status"], "interrupted")
            self.assertEqual(database.get(ai_store.AgentProposal, proposal["id"]).status, "pending")
        self.assertEqual(self.quantities()[1], 5)

    def test_status_has_only_jev_mistral_and_benchmark_route_removed(self):
        status = self.client.get("/api/ai/status").json()
        self.assertEqual({p["name"] for p in status["providers"]}, {"jev", "mistral"})
        self.assertEqual(self.client.post("/api/ai/benchmark", json={}).status_code, 404)
        self.assertEqual(self.client.post("/api/ai/chat", json={"message": " "}).status_code, 422)

    def test_concurrent_chat_gate(self):
        gate.acquire()
        try:
            self.assertEqual(self.client.post("/api/ai/chat", json={"message": "List items"}).status_code, 429)
        finally:
            gate.release()

    def test_inventory_and_manual_price_controls_still_work(self):
        self.assertEqual(self.client.get("/api/inventory?page_size=1").json()["total"], 4)
        self.assertEqual(self.client.get("/api/dashboard").status_code, 200)
        self.assertEqual(self.client.patch("/api/inventory/1/quantity", json={"stock_quantity": 7}).status_code, 200)
        response = self.client.patch("/api/inventory/bulk-price", json={
            "item_ids": [1, 2], "adjustment_type": "percent", "adjustment": 10,
        })
        self.assertEqual(response.status_code, 200)
        with Session(self.engine) as database:
            self.assertEqual(database.get(Inventory, 1).price, 11)
        self.assertEqual(self.quantities()[1], 7)

    def test_startup_and_disabled_ai(self):
        with patch("backend.main.engine", self.engine), TestClient(app):
            self.assertEqual(self.quantities()[1], 5)
        with patch.dict(os.environ, {"AI_ENABLED": "false"}):
            settings.cache_clear()
            try:
                self.assertFalse(self.client.get("/api/ai/status").json()["enabled"])
                self.assertEqual(self.client.post("/api/ai/chat", json={"message": "Show items"}).status_code, 503)
                self.assertEqual(self.client.get("/api/inventory").status_code, 200)
            finally:
                settings.cache_clear()

    def test_local_origin_guard(self):
        for client_host, host, origin in [
            ("192.0.2.1", "localhost", None), ("127.0.0.1", "attacker.example", None),
            ("127.0.0.1", "localhost", "https://attacker.example"),
        ]:
            headers = [(b"host", host.encode())]
            if origin:
                headers.append((b"origin", origin.encode()))
            request = Request({
                "type": "http", "scheme": "http", "path": "/api/ai/chat", "query_string": b"",
                "headers": headers, "client": (client_host, 1), "server": ("localhost", 8000),
            })
            with self.assertRaises(HTTPException):
                require_local(request)

    @patch("backend.services.ai_providers.request")
    def test_jev_typed_router_and_probability_validation(self, request):
        request.return_value = ai_providers.ProviderResult({
            "answers": {"agent": {"choice": "sql", "confidence": .9,
                                   "probabilities": {key: float(key == "sql") for key in ROUTES}}},
        }, "jev-test", 1, Usage())
        decision, _ = ai_providers.route("Show quantity")
        self.assertEqual(decision.agent, "sql")
        self.assertEqual(request.call_args.args[1]["questions"]["agent"]["type"], "choice")
        request.return_value.payload["answers"]["agent"]["probabilities"] = {"sql": 1}
        with self.assertRaises(ai_providers.ProviderError):
            ai_providers.route("Show quantity")

    def parse_events(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        return [
            {"event": frame.splitlines()[0].removeprefix("event: "),
             "data": json.loads(frame.split("\ndata: ", 1)[1])}
            for frame in response.text.split("\n\n") if frame.startswith("event:")
        ]

    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_sse_streams_explanation_and_persists_real_usage_and_stages(self, route, generate, explain):
        route.return_value = routed()
        generate.return_value = SQLPlan(sql="SELECT item_name FROM inventory WHERE item_id = 1"), measured()
        explain.return_value = iter(["## Result\n\n", "**Demo drill**", measured()])
        events = self.parse_events(self.client.post("/api/ai/chat/stream", json={"message": "Show item 1"}))
        self.assertEqual(events[0]["event"], "progress")
        self.assertEqual([e["data"]["text"] for e in events if e["event"] == "delta"], ["## Result\n\n", "**Demo drill**"])
        done = events[-1]["data"]
        self.assertEqual(events[-1]["event"], "done")
        self.assertEqual(done["explanation"], "## Result\n\n**Demo drill**")
        self.assertEqual(done["explanation_status"], "completed")
        self.assertEqual(len(done["usage"]), 3)
        self.assertIn("stream_explanation", [stage["name"] for stage in done["stages"]])
        self.assertEqual(self.client.get(f"/api/ai/runs/{done['id']}").json()["explanation"], done["explanation"])
        facts = explain.call_args.args[1]
        self.assertNotIn("sql", facts)
        self.assertEqual(facts["result"]["rows"], [["Demo drill"]])
        self.assertEqual(self.quantities()[1], 5)
        self.assertTrue(gate.acquire(blocking=False))
        gate.release()

    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_stream_failure_keeps_committed_preview_and_reports_incomplete_explanation(self, route, generate, explain):
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 80)).updates), measured()

        def failed_stream(*args):
            yield "A partial explanation"
            raise ai_providers.ProviderError("Explanation connection timed out")

        explain.side_effect = failed_stream
        events = self.parse_events(self.client.post("/api/ai/chat/stream", json={"message": "Set item 1 to 80"}))
        result = events[-1]["data"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["explanation_status"], "failed")
        self.assertTrue(any(event["event"] == "warning" for event in events))
        self.assertEqual(result["proposals"][0]["status"], "pending")
        self.assertEqual(self.quantities()[1], 5)
        saved = self.client.get(f"/api/ai/runs/{result['id']}").json()
        self.assertEqual(len(saved["proposals"]), 1)
        self.assertIn("timed out", saved["explanation_error"])

    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_cancelled_generator_persists_partial_explanation_without_applying_quantity(self, route, generate, explain):
        route.return_value = routed("quantity")
        generate.return_value = QuantityPlan(updates=quantity_action((1, 80)).updates), measured()
        explain.return_value = iter(["Preview only", "still pending", measured()])
        with Session(self.engine) as database:
            events = ai_agent.chat_events(database, "Set item 1 to 80")
            run_id = None
            for event in events:
                if event["event"] == "validated":
                    run_id = event["data"]["id"]
                if event["event"] == "delta":
                    break
            events.close()
        result = self.client.get(f"/api/ai/runs/{run_id}").json()
        self.assertEqual(result["explanation_status"], "interrupted")
        self.assertEqual(result["explanation"], "Preview only")
        self.assertEqual(result["proposals"][0]["status"], "pending")
        self.assertEqual(self.quantities()[1], 5)

    @patch("backend.services.ai_providers.route")
    def test_cancel_before_agent_dispatch_persists_failed_run(self, route):
        route.return_value = routed("quantity")
        with Session(self.engine) as database:
            events = ai_agent.chat_events(database, "Set item 1 to 80")
            next(events)
            next(events)
            events.close()
            record = database.scalars(select(ai_store.AgentRun)).one()
            self.assertEqual(record.status, "failed")
            self.assertEqual(json.loads(record.result)["proposals"], [])
        self.assertEqual(self.quantities()[1], 5)

    @patch("backend.services.ai_providers.explain")
    @patch("backend.services.ai_providers.route")
    def test_stream_gate_and_clarification_do_not_call_explanation(self, route, explain):
        gate.acquire()
        try:
            self.assertEqual(self.client.post("/api/ai/chat/stream", json={"message": "Show items"}).status_code, 429)
        finally:
            gate.release()
        route.return_value = routed("unsupported")
        events = self.parse_events(self.client.post("/api/ai/chat/stream", json={"message": "Purchase goods"}))
        self.assertEqual(events[-1]["data"]["status"], "needs_clarification")
        explain.assert_not_called()

    def test_explanation_data_budget_marks_incomplete_samples(self):
        result = {
            "agent": "sql", "answer": "Returned 30 rows.", "proposals": [],
            "query": {"columns": ["name"], "rows": [["x"] for _ in range(30)], "row_count": 30, "truncated": False},
        }
        facts = ai_agent.explanation_facts(result)
        self.assertEqual(len(facts["result"]["rows"]), 20)
        self.assertTrue(facts["result"]["incomplete"])
        result["query"]["rows"] = [["x" * 17000]]
        self.assertEqual(ai_agent.explanation_facts(result)["result"]["rows"], [])

    @patch("backend.services.ai_tracing.langsmith_client")
    @patch("backend.services.ai_providers.generate")
    @patch("backend.services.ai_providers.route")
    def test_langsmith_receives_only_metadata_and_parent_child_relationships(self, route, generate, client):
        route.return_value = routed()
        generate.return_value = SQLPlan(sql="SELECT item_name FROM inventory WHERE item_id = 1"), measured()
        config = replace(settings(), langsmith_enabled=True, langsmith_key="private-key-test")
        with patch("backend.services.ai_tracing.settings", return_value=config):
            result = self.client.post("/api/ai/chat", json={"message": "private-user-question"}).json()
        self.assertEqual(result["status"], "completed")
        self.assertIsNotNone(result["observability"]["langsmith_run_id"])
        creates = client.return_value.create_run.call_args_list
        self.assertGreaterEqual(len(creates), 3)
        root_id = creates[0].kwargs["id"]
        self.assertIsNone(creates[0].kwargs["parent_run_id"])
        self.assertEqual(creates[1].kwargs["parent_run_id"], root_id)
        for call in creates:
            self.assertEqual(call.kwargs["inputs"], {})
        for call in client.return_value.update_run.call_args_list:
            self.assertEqual(call.kwargs["outputs"], {})
        exported = str(client.mock_calls)
        for sensitive in ("private-user-question", "Demo drill", "SELECT item_name", "private-key-test"):
            self.assertNotIn(sensitive, exported)

    @patch("backend.services.ai_tracing.langsmith_client")
    def test_langsmith_export_errors_are_explicit_and_do_not_leak_exception_content(self, client):
        from langsmith.utils import LangSmithError

        client.return_value.create_run.side_effect = LangSmithError("sensitive response should not be logged")
        config = replace(settings(), langsmith_enabled=True, langsmith_key="test")
        with patch("backend.services.ai_tracing.settings", return_value=config), ai_tracing.capture() as log:
            with self.assertLogs("backend.services.ai_tracing", level="WARNING") as messages:
                with ai_tracing.span("metadata_only"):
                    pass
        self.assertIn("LangSmith export failed", log.warnings[0])
        self.assertNotIn("sensitive response", str(messages.output))

    def test_actual_langsmith_sdk_serializes_only_explicit_metadata(self):
        from langsmith import Client
        from uuid import uuid4
        from urllib3.util.retry import Retry

        client = Client(
            api_url="https://api.smith.langchain.com", api_key="synthetic-test-only",
            auto_batch_tracing=False, hide_inputs=True, hide_outputs=True,
            omit_traced_runtime_info=True, retry_config=Retry(total=0),
        )
        try:
            with patch.object(Client, "request_with_retries") as send:
                client.create_run(
                    "synthetic-stage", {}, "chain", id=str(uuid4()),
                    project_name="synthetic-test", parent_run_id=str(uuid4()),
                    extra={"metadata": {"agent": "sql"}},
                )
                payload = json.loads(send.call_args.kwargs["request_kwargs"]["data"])
                self.assertEqual(payload["inputs"], {})
                self.assertEqual(payload["extra"].get("runtime", {}), {})
                self.assertEqual(payload["extra"]["metadata"], {"agent": "sql"})
                self.assertTrue(payload["parent_run_id"])
                self.assertNotIn("synthetic-test-only", json.dumps(payload))
        finally:
            client.close(timeout=2)

    def test_mistral_actual_sse_parsing_and_truncation_detection(self):
        actual_client = httpx.Client
        for finish, succeeds in [("stop", True), ("length", False), (None, False)]:
            with self.subTest(finish=finish):
                chunks = [
                    {"model": "stream-test", "choices": [{"delta": {"content": "## Result"}}]},
                    {"model": "stream-test", "choices": [{"delta": {"content": "\n\n**5** units"}, "finish_reason": finish}],
                     "usage": {"prompt_tokens": 25, "completion_tokens": 8}},
                ]
                content = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                transport = httpx.MockTransport(lambda request: httpx.Response(200, text=content))
                with patch("httpx.Client", side_effect=lambda **kwargs: actual_client(transport=transport, **kwargs)):
                    if succeeds:
                        values = list(ai_providers.explain("Synthetic question", {"validated_answer": "5 units"}))
                        self.assertEqual(values[:2], ["## Result", "\n\n**5** units"])
                        self.assertEqual(values[-1].usage.output_tokens, 8)
                        self.assertEqual(values[-1].model, "stream-test")
                    else:
                        with self.assertRaises(ai_providers.ProviderError):
                            list(ai_providers.explain("Synthetic question", {}))

    def test_run_history_contains_chart_metrics_and_ignores_legacy_format(self):
        with Session(self.engine) as database:
            database.add(ai_store.AgentRun(id="compact-format", created_at=1, status="completed", result=json.dumps({
                "schema_version": 2, "agent": "faq", "usage": [], "latency_ms": 123,
                "stages": [
                    {"name": "jev_router", "duration_ms": 10, "status": "OK", "attributes": {"private": "not exported"}},
                    {"name": "faq_agent", "duration_ms": 100, "status": "OK"},
                    {"name": "stream_explanation", "duration_ms": 13, "status": "OK"},
                ],
            }, separators=(",", ":"))))
            database.add(ai_store.AgentRun(id="no-stages", created_at=0, status="completed", result='{"schema_version":2}'))
            database.add(ai_store.AgentRun(id="legacy", created_at=2, status="completed", result='{"schema_version":1}'))
            database.commit()
        runs = self.client.get("/api/ai/runs").json()
        self.assertEqual([run["id"] for run in runs], ["compact-format", "no-stages"])
        self.assertEqual(runs[0]["latency_ms"], 123)
        self.assertEqual(runs[0]["agent"], "faq")
        self.assertEqual(runs[0]["stages"], [
            {"name": "jev_router", "duration_ms": 10, "status": "OK"},
            {"name": "faq_agent", "duration_ms": 100, "status": "OK"},
        ])
        self.assertEqual(runs[1]["stages"], [])

    @patch("backend.services.ai_providers.time.sleep")
    @patch("httpx.Client")
    def test_mistral_bounded_503_retries_and_text_normalization(self, client, sleep):
        post = client.return_value.__enter__.return_value.post
        post.side_effect = [
            httpx.Response(503), httpx.Response(503), httpx.Response(200, json={
                "model": "mistral-test", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"finish_reason": "stop", "message": {"content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": "private-thought"}]},
                    {"type": "text", "text": '{"sql":"SELECT COUNT(*) FROM inventory"}'},
                ]}}],
            }),
        ]
        plan, result = ai_providers.generate("Read only", "Count inventory", SQLPlan)
        self.assertEqual(plan.sql, "SELECT COUNT(*) FROM inventory")
        self.assertEqual(result.attempts, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])
        self.assertNotIn("private-thought", json.dumps(result.payload))
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "none")
        self.assertEqual(result.usage.input_tokens, 10)

    @patch("backend.services.ai_providers.time.sleep")
    @patch("httpx.Client")
    def test_provider_errors_sanitized_and_retries_limited(self, client, sleep):
        post = client.return_value.__enter__.return_value.post
        for provider, status, attempts in [("mistral", 401, 1), ("mistral", 503, 3), ("jev", 503, 1)]:
            with self.subTest(provider=provider, status=status):
                post.reset_mock()
                post.return_value = httpx.Response(status, text="private upstream secret")
                with self.assertRaises(ai_providers.ProviderError) as caught:
                    ai_providers.request(settings().providers[provider], {})
                self.assertNotIn("private upstream secret", str(caught.exception))
                self.assertEqual(post.call_count, attempts)


class ComparisonMetricTests(unittest.TestCase):
    def sample(self, jev_ms, mistral_ms, jev_agent="sql", mistral_agent="sql", **changes):
        row = {
            provider: {
                "status": "ok", "agent": agent, "latency_ms": latency,
                "input_tokens": 100, "output_tokens": 10,
            }
            for provider, latency, agent in (("jev", jev_ms, jev_agent), ("mistral", mistral_ms, mistral_agent))
        }
        for provider, values in changes.items():
            row[provider].update(values)
        return row

    def test_matched_latency_error_exclusion_median_and_agreement(self):
        metrics = ai_comparison.summarize([
            self.sample(10, 100), self.sample(40, 400, mistral_agent="faq"),
            self.sample(20, 200), self.sample(30, 300),
            self.sample(100000, 2, mistral={"status": "error"}),
            self.sample(1, 100000, jev={"status": "error"}),
        ])
        jev, mistral = metrics["providers"]["jev"], metrics["providers"]["mistral"]
        self.assertEqual((jev["mean_ms"], jev["median_ms"], jev["p95_ms"]), (25, 25, 40))
        self.assertEqual((mistral["mean_ms"], mistral["median_ms"], mistral["p95_ms"]), (250, 250, 400))
        self.assertEqual((jev["attempted"], jev["errors"], jev["paired_samples"]), (6, 1, 4))
        self.assertEqual((mistral["attempted"], mistral["errors"], mistral["paired_samples"]), (6, 1, 4))
        self.assertEqual(metrics["agreement_rate"], .75)

    def test_p95_uses_nearest_rank_for_twenty_samples(self):
        metrics = ai_comparison.summarize([self.sample(index, index * 2) for index in range(1, 21)])
        self.assertEqual(metrics["providers"]["jev"]["p95_ms"], 19)
        self.assertEqual(metrics["providers"]["mistral"]["p95_ms"], 38)

    def test_token_means_require_matched_reports_and_preserve_zero(self):
        metrics = ai_comparison.summarize([
            self.sample(10, 20, jev={"input_tokens": 0, "output_tokens": 0}),
            self.sample(10, 20, jev={"input_tokens": None, "output_tokens": None}),
            self.sample(10, 20, mistral={"output_tokens": None}),
        ])
        jev, mistral = metrics["providers"]["jev"], metrics["providers"]["mistral"]
        self.assertEqual(jev["token_samples"], 1)
        self.assertEqual(mistral["token_samples"], 1)
        self.assertEqual(jev["reported_token_samples"], 2)
        self.assertEqual(mistral["reported_token_samples"], 2)
        self.assertEqual(jev["mean_input_tokens"], 0)
        self.assertEqual(jev["mean_output_tokens"], 0)
        self.assertEqual(mistral["mean_input_tokens"], 100)
        unknown = ai_comparison.summarize([
            self.sample(10, 20, jev={"input_tokens": None}),
        ])
        self.assertIsNone(unknown["providers"]["jev"]["mean_input_tokens"])
        self.assertIsNone(unknown["providers"]["mistral"]["mean_input_tokens"])
        self.assertEqual(unknown["providers"]["mistral"]["reported_token_samples"], 1)

    def test_empty_and_all_failed_pairs_do_not_become_zero_latency_or_agreement(self):
        for rows in ([], [self.sample(10, 20, jev={"status": "error"})]):
            metrics = ai_comparison.summarize(rows)
            self.assertEqual(metrics["paired_samples"], 0)
            self.assertIsNone(metrics["agreement_rate"])
            self.assertIsNone(metrics["providers"]["jev"]["mean_ms"])
            self.assertIsNone(metrics["providers"]["mistral"]["median_ms"])
            self.assertIsNone(metrics["providers"]["mistral"]["mean_output_tokens"])


if __name__ == "__main__":
    unittest.main()
