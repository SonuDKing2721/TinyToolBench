import csv
import json
from collections import Counter

from run_benchmark import normalize

INPUT_FILE = "results/results_v3_corrected.csv"
REASONS_FILE = "results/results_v3_with_reasons.csv"
BREAKDOWN_FILE = "results/failure_breakdown_v3.csv"

with open("tasks.json", encoding="utf-8-sig") as file:
    tasks = {task["id"]: task for task in json.load(file)}

with open(INPUT_FILE, newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise SystemExit(f"No rows found in {INPUT_FILE}")

def classify(row):
    status = (row.get("status") or "ok").strip().lower()

    # Keep run errors separate from model mistakes.
    if status in ("error", "timeout"):
        return status

    if row["passed"].strip().lower() == "true":
        return "pass"

    text = row.get("final_text", "")
    expected = tasks[row["task_id"]]["expected_calls"]
    calls = json.loads(row.get("calls") or "[]")

    if not calls:
        if '"name"' in text or '"arguments"' in text:
            return "tool_call_written_as_text"
        return "no_tool_call"

    if not expected:
        return "unexpected_tool_call"

    if len(calls) != len(expected):
        return "wrong_number_of_calls"

    if sorted(call["name"] for call in calls) != sorted(
        call["name"] for call in expected
    ):
        return "wrong_tool"

    actual_shapes = sorted(
        (call["name"], tuple(sorted(call["args"].keys())))
        for call in calls
    )
    expected_shapes = sorted(
        (call["name"], tuple(sorted(call["args"].keys())))
        for call in expected
    )

    if actual_shapes != expected_shapes:
        return "bad_schema"

    if sorted(normalize(call) for call in calls) != sorted(
        normalize(call) for call in expected
    ):
        return "wrong_arguments"

    return "wrong_final_answer"


counts = Counter()
fields = list(rows[0].keys()) + ["reason"]

with open(REASONS_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for row in rows:
        reason = classify(row)
        writer.writerow({**row, "reason": reason})

        if reason != "pass":
            counts[(row["model"], tasks[row["task_id"]]["type"], reason)] += 1

with open(BREAKDOWN_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["model", "task_type", "reason", "count"])

    for (model, task_type, reason), count in sorted(counts.items()):
        writer.writerow([model, task_type, reason, count])

print(f"Created {REASONS_FILE}")
print(f"Created {BREAKDOWN_FILE}")