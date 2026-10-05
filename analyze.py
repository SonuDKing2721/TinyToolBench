import csv
from collections import defaultdict

import matplotlib.pyplot as plt

by_model = defaultdict(list)

with open("results/results.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        by_model[row["model"]].append(row)

# Exclude the extra earlier run so the 1.7B comparison uses 36 attempts.
if len(by_model["qwen3:1.7b"]) > 36:
    by_model["qwen3:1.7b"] = by_model["qwen3:1.7b"][-36:]

models = []
average_speed = []
peak_vram = []

for model, rows in by_model.items():
    models.append(model)
    average_speed.append(
        sum(float(row["tokens_per_s"]) for row in rows) / len(rows)
    )
    peak_vram.append(
        max(float(row["vram_mb"]) for row in rows)
    )

plt.figure(figsize=(9, 5))
plt.bar(models, average_speed)
plt.ylabel("Tokens per second")
plt.title("Average generation speed")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("results/speed.png")
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(models, peak_vram)
plt.axhline(4096, linestyle="--", color="red", label="4 GB GPU capacity")
plt.ylabel("GPU memory (MB)")
plt.title("Highest sampled GPU memory")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("results/vram.png")
plt.close()

print("Created results/speed.png")
print("Created results/vram.png")