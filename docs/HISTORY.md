# Historical Results

This page preserves the original v3 benchmark record. V3 used older scoring rules and task definitions, so its scores are not directly comparable to v4.

## V3 setup

| Item | Value |
|---|---|
| GPU / VRAM | NVIDIA RTX 2050, 4 GB |
| System RAM | 8 GB |
| OS | Windows, PowerShell |
| Ollama CLI version | `0.40.3`, observed 2026-10-19 |
| Python Ollama client | `0.6.3` |
| Temperature | 0 |
| Qwen3 thinking | Off (`think=False`) |
| Runs per task | 3 |
| Historical result-file date | Files last updated 2026-10-19; exact inference dates were not recorded |

All four model tags were reported as Q4_K_M by `ollama show` during the historical run: `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`, and `qwen2.5-coder:7b`.

## V3 results (old scoring)

| Model | Tasks passed | Pass rate | 95% Wilson interval | Avg tokens/s | Maximum sampled post-task GPU memory above idle |
|---|---:|---:|---:|---:|---:|
| `qwen3:4b` | 21/30 | 70.0% | 52-83% | 13.4 | 2335 MB |
| `qwen3:8b` | 19/30 | 63.3% | 46-78% | 4.5 | 2335 MB |
| `qwen3:1.7b` | 12/30 | 40.0% | 25-58% | 78.0 | 1727 MB |
| `qwen2.5-coder:7b` | 5/30 | 16.7% | 7-34% | 5.1 | 2265 MB |

The old CSV's BOM-related model-label correction assigned 90 blank model labels to `qwen3:1.7b`; that attribution correction did not change any run-level pass/fail values. Comparing majority-task counts, the 4B moved from 21 to 17, the 8B from 19 to 20, the 1.7B from 12 to 11, and the coder from 5 to 2. These differences cannot be attributed to any one specific fix: v4 now requires meaningful no-tool answers, checks numeric/text answers more carefully, enforces order on dependent chains, corrects x7's arguments/category, and replaces subjective x12 with a fixed-note lookup.

The historical no-tool tasks rewarded an empty response, producing a never-calls baseline of 5/30. Under v4, a model that never calls a tool and gives no answer passes 0/30; the old baseline does not apply.

Historical CPU/GPU placement was captured from one `ollama ps` snapshot per model and can vary:

| Model | CPU / GPU | Historical avg tokens/s |
|---|---:|---:|
| `qwen3:1.7b` | 0% / 100% | 78.0 |
| `qwen3:4b` | 33% / 67% | 13.4 |
| `qwen2.5-coder:7b` | 55% / 45% | 5.1 |
| `qwen3:8b` | 61% / 39% | 4.5 |
