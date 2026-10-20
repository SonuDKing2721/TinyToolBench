import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_benchmark
from run_benchmark import read_csv_rows, score, validate_tasks
from tools import NOTES


class BenchmarkTests(unittest.TestCase):
    def test_all_task_definitions_match_registered_tool_schemas(self):
        tasks = json.loads(Path("tasks.json").read_text(encoding="utf-8-sig"))

        validate_tasks(tasks)

    def test_task_validation_rejects_string_for_integer_tool_argument(self):
        tasks = json.loads(Path("tasks.json").read_text(encoding="utf-8-sig"))
        task = next(task for task in tasks if task["id"] == "x7")
        task["expected_calls"][1]["args"]["a"] = "8"

        with self.assertRaisesRegex(ValueError, "integer"):
            validate_tasks(tasks)

    def test_x12_checks_a_fixed_note_instead_of_a_subjective_opinion(self):
        tasks = json.loads(Path("tasks.json").read_text(encoding="utf-8-sig"))
        task = next(task for task in tasks if task["id"] == "x12")

        self.assertEqual(task["type"], "selection")
        self.assertEqual(
            task["expected_calls"],
            [{"name": "search_notes", "args": {"keyword": "football"}}],
        )
        self.assertEqual(task["expected_answer"], "Lionel Messi")
        self.assertIn("Lionel Messi", NOTES["football"])

    def test_writes_the_full_model_response_to_csv(self):
        final_text = "x" * 250

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                Path("tasks.json").write_text(
                    json.dumps([{
                        "id": "long-answer",
                        "type": "single",
                        "prompt": "Return a long answer.",
                        "expected_calls": [],
                    }]),
                    encoding="utf-8",
                )
                with (
                    patch.object(run_benchmark, "SOURCE_FILE", "results/old.csv"),
                    patch.object(run_benchmark, "OUT_FILE", "results/test.csv"),
                    patch.object(run_benchmark, "run_task", return_value=([], final_text, 1.0, 2.0)),
                    patch.object(run_benchmark, "post_task_vram_mb", return_value=0),
                    patch.object(run_benchmark, "write_run_metadata", return_value=("test-session", "metadata.json")),
                    patch.object(sys, "argv", ["run_benchmark.py", "test-model"]),
                    redirect_stdout(io.StringIO()),
                ):
                    run_benchmark.main()
                rows = read_csv_rows("results/test.csv")
            finally:
                os.chdir(previous_directory)

        self.assertEqual(rows[0]["final_text"], final_text)
        self.assertEqual(rows[0]["session_id"], "test-session")

    def test_run_metadata_records_command_versions_and_model_quantization(self):
        model_details = SimpleNamespace(
            family="qwen3",
            parameter_size="2.0B",
            quantization_level="Q4_K_M",
        )

        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                Path("tasks.json").write_text("[]", encoding="utf-8")
                Path("tools.py").write_text("# tools", encoding="utf-8")
                with (
                    patch.object(run_benchmark, "ollama_cli_version", return_value="ollama version is 0.40.3"),
                    patch.object(run_benchmark, "ollama_python_client_version", return_value="0.6.3"),
                    patch.object(run_benchmark.ollama, "show", return_value=SimpleNamespace(details=model_details)),
                    patch.object(sys, "argv", ["run_benchmark.py", "qwen3:1.7b"]),
                ):
                    session_id, metadata_path = run_benchmark.write_run_metadata(["qwen3:1.7b"])
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            finally:
                os.chdir(previous_directory)

        self.assertEqual(metadata["session_id"], session_id)
        self.assertEqual(metadata["command"][-1], "qwen3:1.7b")
        self.assertEqual(metadata["ollama_python_client_version"], "0.6.3")
        self.assertEqual(metadata["ollama_cli_version"], "ollama version is 0.40.3")
        self.assertEqual(metadata["models"]["qwen3:1.7b"]["quantization_level"], "Q4_K_M")

    def test_x7_matches_its_search_then_multiply_chain(self):
        tasks = json.loads(Path("tasks.json").read_text(encoding="utf-8-sig"))
        task = next(task for task in tasks if task["id"] == "x7")
        calls = task["expected_calls"]

        self.assertEqual(task["type"], "chain")
        self.assertIn("search my notes", task["prompt"].casefold())
        self.assertEqual(calls[0]["name"], "search_notes")
        self.assertEqual(calls[1]["name"], "multiply")
        self.assertEqual(calls[1]["args"], {"a": 8, "b": 6})
        self.assertTrue(score(task, calls, "There are 48 books."))

    def test_ordered_chain_rejects_calls_in_the_wrong_order(self):
        expected_calls = [
            {"name": "add", "args": {"a": 10, "b": 5}},
            {"name": "multiply", "args": {"a": 15, "b": 3}},
        ]
        task = {
            "type": "chain",
            "ordered": True,
            "expected_calls": expected_calls,
            "expected_answer": "45",
        }
        reversed_calls = list(reversed(expected_calls))

        self.assertFalse(score(task, reversed_calls, "The answer is 45."))

    def test_unordered_calls_remain_flexible_when_order_is_not_required(self):
        expected_calls = [
            {"name": "get_price", "args": {"item": "keyboard"}},
            {"name": "get_price", "args": {"item": "mouse"}},
        ]
        task = {
            "type": "custom",
            "ordered": False,
            "expected_calls": expected_calls,
        }

        self.assertTrue(score(task, list(reversed(expected_calls)), "done"))

    def test_reads_model_header_when_csv_has_utf8_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=["model", "task_id"])
                writer.writeheader()
                writer.writerow({"model": "qwen3:1.7b", "task_id": "s1"})

            rows = read_csv_rows(path)

        self.assertEqual(rows[0]["model"], "qwen3:1.7b")

    def test_no_tool_task_does_not_pass_with_an_empty_answer(self):
        task = {"type": "no_tool", "expected_calls": []}

        self.assertFalse(score(task, [], ""))

    def test_no_tool_task_can_pass_with_an_answer_and_no_tool_calls(self):
        task = {
            "type": "no_tool",
            "expected_calls": [],
            "expected_answer": "Paris",
        }

        self.assertTrue(score(task, [], "The capital of France is Paris."))

    def test_numeric_answer_does_not_match_a_larger_number(self):
        task = {"type": "single", "expected_calls": [], "expected_answer": "4"}

        self.assertFalse(score(task, [], "The answer is 14."))

    def test_numeric_answer_matches_a_standalone_formatted_number(self):
        task = {
            "type": "single",
            "expected_calls": [],
            "expected_answer": "1200",
        }

        self.assertTrue(score(task, [], "The total is 1,200."))

    def test_text_answer_ignores_case_and_punctuation(self):
        task = {
            "type": "no_tool",
            "expected_calls": [],
            "expected_answer": "Lionel Messi",
        }

        self.assertTrue(score(task, [], "I think lionel messi is the GOAT."))
