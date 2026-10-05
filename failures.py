import csv
import json
from collections import Counter, defaultdict

from run_benchmark import normalize

with open("tasks.json", encoding="utf-8") as f:
    tasks = {task["id"]: task for task in json.load(f)}

with open("results/results.csv", newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

# qwen3:1.7b has one extra earlier run; use its latest 36 attempts for fairness.
by_model = defaultdict(list)
for row in all_rows:
    by_model[row["model"]].append(row)

rows = []
for model, model_rows in by_model.items():
    if model == "qwen3:1.7b" and len(model_rows) > 36:
        model_rows = model_rows[-36:]
    rows.extend(model_rows)


def classify(row):
    if row["passed"].strip().lower() == "true":
        return "pass"

    text = row["final_text"]
    if text.startswith("ERROR"):
        return "runtime_error"

    expected = tasks[row["task_id"]]["expected_calls"]
    calls = json.loads(row["calls"] or "[]")

    if not calls:
        if '"name"' in text or '"arguments"' in text:
            return "tool_call_written_as_text"
        return "no_tool_call"

    if not expected:
        return "unexpected_tool_call"

    if len(calls) != len(expected):
        return "wrong_number_of_calls"

    if sorted(c["name"] for c in calls) != sorted(e["name"] for e in expected):
        return "wrong_tool"

    if sorted(normalize(c) for c in calls) != sorted(normalize(e) for e in expected):
        return "wrong_arguments"

    return "wrong_final_answer"


breakdown = Counter()

with open("results/results_with_reasons.csv", "w", newline="", encoding="utf-8") as f:
    fields = list(rows[0].keys()) + ["reason"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for row in rows:
        reason = classify(row)
        writer.writerow({**row, "reason": reason})

        if reason != "pass":
            task_type = tasks[row["task_id"]]["type"]
            breakdown[(row["model"], task_type, reason)] += 1

with open("results/failure_breakdown.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "task_type", "reason", "count"])

    for (model, task_type, reason), count in sorted(breakdown.items()):
        writer.writerow([model, task_type, reason, count])

print("Created results/results_with_reasons.csv")
print("Created results/failure_breakdown.csv")