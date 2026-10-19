import csv
from collections import defaultdict

INPUT_FILE = "results/results_v3_corrected.csv"

passed = defaultdict(int)
total = defaultdict(int)

with open(INPUT_FILE, newline="", encoding="utf-8-sig") as file:
    for row in csv.DictReader(file):
        if row.get("status") != "ok":
            continue

        key = (row["model"], row["type"])
        total[key] += 1
        if row["passed"].strip().lower() == "true":
            passed[key] += 1

print(f"{'MODEL':<20} {'TASK TYPE':<12} {'PASSED':>8} {'TOTAL':>7} {'RATE':>8}")
print("-" * 60)

for model, task_type in sorted(total):
    count = passed[(model, task_type)]
    attempts = total[(model, task_type)]
    rate = count / attempts * 100
    print(f"{model:<20} {task_type:<12} {count:>4}/{attempts:<3} {rate:>7.1f}%")

    