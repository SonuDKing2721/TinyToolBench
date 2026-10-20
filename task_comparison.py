import csv
import json
import math
from collections import defaultdict
from itertools import combinations

INPUT_FILE = "results/results_v4.csv"
RUNS_PER_TASK = 3


def wilson_interval(successes, total):
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return (
        100 * (center - margin),
        100 * (center + margin),
    )


with open("tasks.json", encoding="utf-8-sig") as file:
    tasks = [task["id"] for task in json.load(file)]

with open(INPUT_FILE, newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

attempts = defaultdict(list)
for row in rows:
    if row.get("status", "ok").strip().lower() == "ok":
        key = (row["model"], row["task_id"])
        attempts[key].append(row["passed"].strip().lower() == "true")

models = sorted({row["model"] for row in rows})
task_results = defaultdict(dict)

print("Task-level results (task passes when at least 2 of 3 runs pass)")
for model in models:
    for task_id in tasks:
        runs = attempts[(model, task_id)]
        if len(runs) != RUNS_PER_TASK:
            print(f"WARNING: {model} {task_id} has {len(runs)} valid runs")
            continue
        task_results[model][task_id] = sum(runs) >= 2

    passed = sum(task_results[model].values())
    total = len(task_results[model])
    low, high = wilson_interval(passed, total)
    print(f"{model}: {passed}/{total} tasks, 95% Wilson interval {low:.1f}%-{high:.1f}%")

print("\nPaired comparison (same tasks; majority result per task)")
for first, second in combinations(models, 2):
    common = set(task_results[first]) & set(task_results[second])
    first_wins = sum(
        task_results[first][task] and not task_results[second][task]
        for task in common
    )
    second_wins = sum(
        task_results[second][task] and not task_results[first][task]
        for task in common
    )
    ties = len(common) - first_wins - second_wins
    print(
        f"{first} vs {second}: "
        f"{first_wins} wins / {ties} ties / {second_wins} losses "
        f"across {len(common)} shared tasks"
    )
