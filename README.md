# TinyToolBench

A small benchmark for comparing local language models on tool-calling tasks through [Ollama](https://ollama.com). It checks whether a model picks the right tool, supplies correct arguments, handles multi-step tasks, and avoids calling a tool when none is needed. It also records generation speed and GPU-memory samples on one 4 GB laptop GPU.

TinyToolBench is a hand-written, fixed benchmark run on one machine. It is useful for exploring behavior on these tasks, not for ranking models generally. The corrected v4 run is complete (360 attempts: 30 tasks x 3 runs x 4 models). The v3 figures below are historical and must not be interpreted as v4 results.

## Setup used for the historical v3 run

| Item | Value |
|---|---|
| GPU / VRAM | NVIDIA RTX 2050, 4 GB |
| System RAM | 8 GB |
| OS | Windows, PowerShell |
| Ollama CLI version | `0.40.3` observed on 2026-10-19; this was the CLI-reported version |
| Python Ollama client | `0.6.3` |
| Temperature | 0 |
| Qwen3 thinking | Off (`think=False` for Qwen3 models) |
| Runs per task | 3 |
| Historical result-file date | Files last updated 2026-10-19; exact inference dates were not recorded |

The v4 runner writes a metadata JSON beside its result CSV with the command, Python and Ollama versions, task/tool file hashes, model family/size/quantization, benchmark settings, and a GPU-memory baseline when available.

### Quantization in the historical run

| Model | Quantization |
|---|---|
| `qwen3:1.7b` | Q4_K_M |
| `qwen3:4b` | Q4_K_M |
| `qwen3:8b` | Q4_K_M |
| `qwen2.5-coder:7b` | Q4_K_M |

These are the values previously observed with `ollama show`; each new v4 run records the model metadata it sees at run time.

## Tasks

There are 30 hand-written tasks in `tasks.json`, run 3 times per model (90 attempts per model). Every task offers the same four fixed-data tools, with no internet access: `add`, `multiply`, `get_price`, and `search_notes`. All four tools are offered each time, so the model must select among distractors.

| Category | Tasks | What it tests |
|---|---:|---|
| Single | 6 | One tool call |
| Selection | 5 | Choose the right tool for the request |
| Chain | 9 | Multiple dependent tool calls in the required order |
| No-tool | 6 | Answer directly without calling a tool |
| Robustness | 4 | Requests that test tool limits |

## Scoring

Scoring is automatic; there is no LLM judge. A run passes only when the model makes exactly the expected calls with the expected arguments. Argument order for `add` and `multiply` does not matter. Calls are order-independent unless a task explicitly has `"ordered": true`; dependent chain tasks use that setting.

If a task defines an `expected_answer`, the final answer must contain the expected number as a whole number token, or the expected text as a case- and punctuation-normalized sequence of words. This is stricter than a substring check but is not semantic evaluation. No-tool tasks must return a non-empty answer; tasks with a gold answer must also match it. A task passes when at least 2 of its 3 attempts pass. The reported 95% Wilson interval is calculated over tasks, not individual attempts.

The task validator checks IDs, categories, prompts, expected calls, tool names, argument names, and argument types before inference begins.

## Corrected v4 results

Each task passes when at least 2 of its 3 runs pass. These are the task-level results under the corrected checker.

| Model | Tasks passed | Pass rate | 95% Wilson interval | Avg tokens/s | Max sampled post-task GPU memory |
|---|---:|---:|---:|---:|---:|
| `qwen3:8b` | 20/30 | 66.7% | 48.8-80.8% | 2.9 | 2335 MB |
| `qwen3:4b` | 17/30 | 56.7% | 39.2-72.6% | 12.1 | 2337 MB |
| `qwen3:1.7b` | 11/30 | 36.7% | 21.9-54.5% | 77.1 | 1727 MB |
| `qwen2.5-coder:7b` | 2/30 | 6.7% | 1.8-21.3% | 5.0 | 2265 MB |

The 8B model had the highest observed task score, but its interval overlaps the 4B model's interval. On paired task outcomes, the 8B won 4 tasks, the 4B won 1, and they tied on 25. With only 30 tasks, this is suggestive, not proof that one model is generally better.

### V4 by category (individual runs)

| Model | Chain (27) | No-tool (18) | Robustness (12) | Selection (15) | Single (18) |
|---|---:|---:|---:|---:|---:|
| `qwen3:8b` | 15/27 | 12/18 | 9/12 | 9/15 | 15/18 |
| `qwen3:4b` | 6/27 | 12/18 | 8/12 | 10/15 | 15/18 |
| `qwen3:1.7b` | 0/27 | 12/18 | 4/12 | 9/15 | 8/18 |
| `qwen2.5-coder:7b` | 0/27 | 6/18 | 0/12 | 0/15 | 0/18 |

The coder model wrote tool-call-shaped text instead of making structured calls in 84 of 90 runs. The 1.7B model made no passing chain runs; the 8B model had the strongest chain result (15/27).

The CSV has 360 unique model/task/run records, all with `status=ok`. The full answers are retained in `results/results_v4.csv`; the run metadata sidecars record two resumed sessions with matching task and tool hashes. The completed run produced `results/performance_summary_v4.csv`, `results/results_v4_with_reasons.csv`, `results/failure_breakdown_v4.csv`, `results/speed_v4.png`, and `results/post_task_vram_v4.png`.

## Historical v3 results (old scoring)

These figures were generated before the v4 scoring and task-definition corrections. They are retained for historical context only; their scores are not directly comparable to v4 because both the checker and two task definitions changed.

| Model | Tasks passed | Pass rate | 95% Wilson interval | Avg tokens/s | Maximum sampled post-task GPU memory above idle |
|---|---:|---:|---:|---:|---:|
| `qwen3:4b` | 21/30 | 70.0% | 52-83% | 13.4 | 2335 MB |
| `qwen3:8b` | 19/30 | 63.3% | 46-78% | 4.5 | 2335 MB |
| `qwen3:1.7b` | 12/30 | 40.0% | 25-58% | 78.0 | 1727 MB |
| `qwen2.5-coder:7b` | 5/30 | 16.7% | 7-34% | 5.1 | 2265 MB |

The old CSV's BOM-related model-label correction assigned 90 blank model labels to `qwen3:1.7b`; that attribution correction did not change any run-level pass/fail values. Comparing majority-task counts, the 4B moved from 21 to 17, the 8B from 19 to 20, the 1.7B from 12 to 11, and the coder from 5 to 2. These differences cannot be attributed to one specific fix: v4 now requires meaningful no-tool answers, checks numeric/text answers more carefully, enforces order on dependent chains, corrects x7's arguments/category, and replaces subjective x12 with a fixed-note lookup.

The historical no-tool tasks rewarded an empty response, producing a never-calls baseline of 5/30. Under v4, a model that never calls a tool and gives no answer passes 0/30; the old baseline does not apply.

Historical CPU/GPU placement was captured from one `ollama ps` snapshot per model and can vary:

| Model | CPU / GPU | Historical avg tokens/s |
|---|---:|---:|
| `qwen3:1.7b` | 0% / 100% | 78.0 |
| `qwen3:4b` | 33% / 67% | 13.4 |
| `qwen2.5-coder:7b` | 55% / 45% | 5.1 |
| `qwen3:8b` | 61% / 39% | 4.5 |

## Measurement notes

GPU memory is sampled after each task with `nvidia-smi`. The v4 charts and summary report the maximum raw post-task sample; this is **not** continuous peak VRAM monitoring and can miss higher usage during inference. The completed v4 data spans two resumed sessions with different idle readings, so raw post-task memory is shown instead of subtracting one baseline from all rows. CPU/GPU placement is a runtime snapshot, not a per-task measurement.

## Limitations

- Only 30 tasks, written by one person; small differences are uncertain.
- All tasks run at temperature 0, so repeated attempts may be very similar and are not independent evidence.
- Results depend on one machine, GPU, runtime, model tags, and tool schemas.
- Quantization is not experimentally isolated; model families, sizes, and tags differ.
- The answer checker is deterministic and more exact than substring matching, but it cannot judge meaning or paraphrases.
- Fake tools are simpler than real tools with network failures, changing data, permissions, or side effects.
- Historical v3 scores used older scoring rules and should not be treated as corrected results.

## Related work

- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) evaluates function calling with a much broader set of tasks and settings. TinyToolBench is a small fixed local suite.
- [BenchLoop](https://github.com/outsourc-e/bench-loop) is a local-first benchmarking project with broader quality, speed, reliability, and multi-turn evaluation capabilities.

TinyToolBench differs in its small fixed set of local tools, deterministic checker without an LLM judge, and focus on speed and memory observations from a constrained laptop.

## Running the benchmark

1. Install [Ollama](https://ollama.com), then pull the models you want to test:
   ```powershell
   ollama pull qwen3:1.7b
   ollama pull qwen3:4b
   ollama pull qwen3:8b
   ollama pull qwen2.5-coder:7b
   ```
2. Create/activate the project virtual environment and install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Run all models, or pass model tags to run only selected models:
   ```powershell
   python run_benchmark.py
   python run_benchmark.py qwen3:4b
   ```
   The corrected raw attempts append to `results/results_v4.csv`. Each invocation also writes `results/run_metadata_<session>.json`. Existing v3 data is never migrated into v4 because it used different scoring rules.
4. Analyze v4 data:
   ```powershell
   python performance_table.py
   python analyze.py
   python failures.py
   python category_table.py
   python task_comparison.py
   ```
   These create the v4 summary CSVs and charts under `results/`. Run `python run_benchmark.py` first; the analysis scripts intentionally do not fall back to historical data.

## License

MIT. See [LICENSE](LICENSE).
