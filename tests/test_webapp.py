import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from webapp import main


class FakeEngine:
    def answer(self, question):
        return main.GraphV2QA.response(
            "domain_answer",
            question,
            "Verified answer.",
            [{
                "chunk_id": "C_0001_001",
                "pdf_page": 1,
                "printed_page": None,
                "evidence": "Verified evidence.",
                "confidence": 0.9,
            }],
            [],
        )


class WebappTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_small_talk_does_not_create_graph_engine(self):
        with patch.object(main, "get_engine") as mocked_engine:
            response = self.client.post("/ask", json={"query": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["kind"], "small_talk")
        mocked_engine.assert_not_called()

    def test_repeated_small_talk_requests_are_independent(self):
        with patch.object(main, "get_engine") as mocked_engine:
            first = self.client.post("/ask", json={"query": "hi"})
            second = self.client.post("/ask", json={"query": "bye"})
            third = self.client.post("/ask", json={"query": "who are you?"})
        self.assertEqual([first.status_code, second.status_code, third.status_code], [200, 200, 200])
        mocked_engine.assert_not_called()

    def test_domain_request_uses_engine(self):
        with patch.object(main, "get_engine", return_value=FakeEngine()):
            response = self.client.post(
                "/ask",
                json={"query": "Which equipment is used for microscopic examination?"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["kind"], "domain_answer")
        self.assertEqual(response.json()["sources"][0]["chunk_id"], "C_0001_001")

    def test_relation_intents(self):
        cases = {
            "Which equipment is used for microscopic examination?": "USES_EQUIPMENT",
            "Which reagent is used for Gram staining?": "USES_REAGENT",
            "How is African trypanosomiasis transmitted?": "TRANSMITTED_BY",
            "What symptoms are associated with visceral leishmaniasis?": "HAS_FINDING",
            "Where are the eggs found?": "FOUND_IN",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(main.relation_intent(question), expected)

    def test_microscopic_and_microscope_normalize_together(self):
        self.assertEqual(
            main.normalize_token("microscopic"),
            main.normalize_token("microscope"),
        )

    def test_fact_answer_is_readable(self):
        rows = [
            {"source_name": "Microscopic examination", "target_name": "microscope"},
            {"source_name": "Microscopic examination", "target_name": "10x objective"},
        ]
        answer = main.GraphV2QA.compose_fact_answer("USES_EQUIPMENT", rows)
        self.assertIn("Microscopic examination uses", answer)
        self.assertIn("microscope", answer)
        self.assertIn("10x objective", answer)

    def test_microscope_question_uses_graph_facts(self):
        engine = main.GraphV2QA.__new__(main.GraphV2QA)
        engine.relation_facts = lambda relation_type: [
            {
                "source_id": "E_PROC",
                "source_name": "Microscopic examination",
                "source_type": "PROCEDURE",
                "relation_type": "USES_EQUIPMENT",
                "target_id": "E_MICROSCOPE",
                "target_name": "microscope",
                "target_type": "EQUIPMENT",
                "chunk_id": "C_0174_002",
                "pdf_page": 174,
                "printed_page": 162,
                "evidence": "Examine the wet preparation under the microscope.",
                "confidence": 0.88,
            }
        ]
        engine.verified_images = lambda question, relation_type, pages: []
        response = engine.answer(
            "Which equipment is used for microscopic examination?"
        )
        self.assertEqual(response["kind"], "domain_answer")
        self.assertIn("microscope", response["answer"])
        self.assertEqual(response["sources"][0]["pdf_page"], 174)

    def test_unrelated_chunk_is_not_grounded(self):
        row = {
            "text": "The laboratory uses sterile glassware.",
            "entity_names": ["glassware"],
        }
        self.assertFalse(
            main.GraphV2QA.chunk_is_grounded(
                "Who is the president of Germany?", row
            )
        )


if __name__ == "__main__":
    unittest.main()
