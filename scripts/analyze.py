import csv
from collections import defaultdict

import matplotlib.pyplot as plt

INPUT_FILE = "results/results_v4.csv"
GPU_CAPACITY_MB = 4096

models = [
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen2.5-coder:7b",
    "qwen3:8b",
]
by_model = defaultdict(list)

with open(INPUT_FILE, newline="", encoding="utf-8-sig") as file:
    for row in csv.DictReader(file):
        if row.get("status", "ok").strip().lower() == "ok":
            by_model[row["model"]].append(row)

average_speed = []
max_post_task_vram = []

for model in models:
    rows = by_model[model]
    if not rows:
        raise SystemExit(f"No successful rows found for {model}")

    speeds = [float(row["tokens_per_s"]) for row in rows]
    vram_values = [
        float(row["post_task_vram_mb"])
        for row in rows
        if row.get("post_task_vram_mb", "").strip()
    ]

    average_speed.append(sum(speeds) / len(speeds))

    max_post_task_vram.append(max(vram_values) if vram_values else 0)

print("Model performance summary")
for index, model in enumerate(models):
    print(
        f"{model}: "
        f"{average_speed[index]:.1f} tokens/s average, "
        f"{max_post_task_vram[index]:.0f} MB maximum sampled post-task GPU memory"
    )

plt.figure(figsize=(9, 5))
plt.bar(models, average_speed)
plt.ylabel("Tokens per second")
plt.title("Average generation speed - expanded benchmark")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("results/speed_v4.png")
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(models, max_post_task_vram)
plt.axhline(
    GPU_CAPACITY_MB,
    linestyle="--",
    color="red",
    label="4 GB GPU capacity",
)
plt.ylabel("Post-task GPU memory (MB)")
plt.title("Maximum sampled post-task GPU memory - corrected benchmark")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("results/post_task_vram_v4.png")
plt.close()

print("Created results/speed_v4.png")
print("Created results/post_task_vram_v4.png")
