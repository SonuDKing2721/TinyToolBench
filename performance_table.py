import csv
import json
import math
from collections import defaultdict

INPUT_FILE = "results/results_v3_corrected.csv"
OUTPUT_FILE = "results/performance_summary_v3.csv"
IDLE_VRAM_MB = 0  # Your idle nvidia-smi reading was 0 MiB.
RUNS_PER_TASK = 3


def wilson_interval(successes, total):
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return 100 * (center - margin), 100 * (center + margin)


with open("tasks.json", encoding="utf-8-sig") as file:
    task_ids = [task["id"] for task in json.load(file)]

with open(INPUT_FILE, newline="", encoding="utf-8-sig") as file:
    all_rows = list(csv.DictReader(file))

by_model = defaultdict(list)
by_model_task = defaultdict(list)

for row in all_rows:
    if row.get("status", "ok").strip().lower() != "ok":
        continue

    by_model[row["model"]].append(row)
    by_model_task[(row["model"], row["task_id"])].append(
        row["passed"].strip().lower() == "true"
    )

summary = []

for model, rows in by_model.items():
    attempts_passed = sum(
        row["passed"].strip().lower() == "true" for row in rows
    )
    attempt_total = len(rows)

    tasks_passed = 0
    for task_id in task_ids:
        runs = by_model_task[(model, task_id)]
        if len(runs) != RUNS_PER_TASK:
            raise SystemExit(
                f"{model} / {task_id} has {len(runs)} valid runs, "
                f"expected {RUNS_PER_TASK}"
            )
        if sum(runs) >= 2:
            tasks_passed += 1

    task_total = len(task_ids)
    ci_low, ci_high = wilson_interval(tasks_passed, task_total)

    speeds = [float(row["tokens_per_s"]) for row in rows]
    vram_values = [
        float(row["vram_mb"])
        for row in rows
        if row.get("vram_mb", "").strip()
    ]
    peak_vram = max(vram_values) - IDLE_VRAM_MB if vram_values else 0

    summary.append({
        "model": model,
        "attempt_passed": attempts_passed,
        "attempt_total": attempt_total,
        "attempt_pass_rate_pct": round(
            100 * attempts_passed / attempt_total, 1
        ),
        "tasks_passed": tasks_passed,
        "task_total": task_total,
        "task_pass_rate_pct": round(100 * tasks_passed / task_total, 1),
        "wilson_95_low_pct": round(ci_low, 1),
        "wilson_95_high_pct": round(ci_high, 1),
        "average_tokens_per_s": round(sum(speeds) / len(speeds), 1),
        "peak_vram_above_idle_mb": round(peak_vram),
    })

summary.sort(key=lambda item: item["task_pass_rate_pct"], reverse=True)

print(
    f"{'Model':<20} {'Attempts':>12} {'Tasks':>10} "
    f"{'Wilson 95%':>17} {'Tok/s':>8} {'GPU MB':>8}"
)
print("-" * 82)

for item in summary:
    attempts = f"{item['attempt_passed']}/{item['attempt_total']}"
    tasks = f"{item['tasks_passed']}/{item['task_total']}"
    interval = (
        f"{item['wilson_95_low_pct']:.1f}-"
        f"{item['wilson_95_high_pct']:.1f}%"
    )
    print(
        f"{item['model']:<20} "
        f"{attempts:>12} "
        f"{tasks:>10} "
        f"{interval:>17} "
        f"{item['average_tokens_per_s']:>8.1f} "
        f"{item['peak_vram_above_idle_mb']:>8}"
    )

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(f"\nSaved spreadsheet-ready table to {OUTPUT_FILE}")