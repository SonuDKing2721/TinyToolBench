import csv
from collections import defaultdict

import matplotlib.pyplot as plt

INPUT_FILE = "results/results_v3_corrected.csv"
IDLE_VRAM_MB = 0  # Your idle nvidia-smi reading showed 0 MiB.
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
peak_vram_above_idle = []

for model in models:
    rows = by_model[model]
    if not rows:
        raise SystemExit(f"No successful rows found for {model}")

    speeds = [float(row["tokens_per_s"]) for row in rows]
    vram_values = [
        float(row["vram_mb"])
        for row in rows
        if row.get("vram_mb", "").strip()
    ]

    average_speed.append(sum(speeds) / len(speeds))

    if vram_values:
        peak_vram_above_idle.append(
            max(0, max(vram_values) - IDLE_VRAM_MB)
        )
    else:
        peak_vram_above_idle.append(0)

print("Model performance summary")
for index, model in enumerate(models):
    print(
        f"{model}: "
        f"{average_speed[index]:.1f} tokens/s average, "
        f"{peak_vram_above_idle[index]:.0f} MB peak GPU memory above idle"
    )

plt.figure(figsize=(9, 5))
plt.bar(models, average_speed)
plt.ylabel("Tokens per second")
plt.title("Average generation speed - expanded benchmark")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("results/speed_v3.png")
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(models, peak_vram_above_idle)
plt.axhline(
    GPU_CAPACITY_MB - IDLE_VRAM_MB,
    linestyle="--",
    color="red",
    label="4 GB GPU capacity",
)
plt.ylabel("GPU memory above idle (MB)")
plt.title("Peak GPU memory - expanded benchmark")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("results/vram_v3.png")
plt.close()

print("Created results/speed_v3.png")
print("Created results/vram_v3.png")