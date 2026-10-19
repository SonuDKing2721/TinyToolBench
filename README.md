# TinyToolBench

A small benchmark for comparing local language models on tool-calling tasks through [Ollama](https://ollama.com). It checks whether a model picks the right tool, supplies correct arguments, handles multi-step tasks, and avoids calling a tool when none is needed. It also records speed and GPU memory on a 4 GB laptop GPU.

This is a small, hand-written benchmark run on one machine. It is meant for comparing models on constrained hardware, not for ranking models in general. See [Limitations](#limitations).

## Setup used for these results

| Item | Value |
|---|---|
| GPU / VRAM | NVIDIA RTX 2050, 4 GB (checked with `nvidia-smi`) |
| System RAM | 8 GB |
| OS | Windows, PowerShell |
| Ollama version | Client `0.40.3` observed on 2026-10-19; the server version used during the benchmark was not recorded |
| Temperature | 0 (confirmed in `run_benchmark.py`) |
| Qwen3 thinking | Off (`think=False` for Qwen3 models; confirmed in `run_benchmark.py`) |
| Runs per task | 3 |
| Date of run | Result files last updated 2026-10-19; the exact run date was not recorded separately |

### Quantization per model

| Model | Quantization |
|---|---|
| `qwen3:1.7b` | Q4_K_M (reported by `ollama show`) |
| `qwen3:4b` | Q4_K_M |
| `qwen3:8b` | Q4_K_M |
| `qwen2.5-coder:7b` | Q4_K_M |

## Tasks

30 hand-written tasks in `tasks.json`, each run 3 times per model (90 runs per model). Every task offers the model the same four fake tools with fixed data (no internet): `add`, `multiply`, `get_price`, `search_notes`. All four are available every time, so the model must choose among distractors.

| Category | Tasks | What it tests |
|---|---:|---|
| Single | 6 | One tool call, e.g. adding two numbers |
| Selection | 5 | Choose the right tool for the request |
| Chain | 8 | Use several tools in the right order |
| No-tool | 7 | Answer directly without calling a tool, including a trap where no tool fits |
| Robustness | 4 | Requests that test the tools' limits |

## Scoring

Scoring is automatic, with no LLM judge. A run passes when:

- the model calls the right tool(s) with the right arguments (for `add` and `multiply`, argument order is ignored);
- for chains, the calls are compared without requiring a particular call order;
- when a task has an expected answer, the final answer contains it (a lenient substring check).

A **task** counts as passed when at least 2 of its 3 runs pass. If temperature is 0, the 3 runs are nearly identical, so the effective sample is the 30 tasks, not 90 runs. Uncertainty is shown as a 95% Wilson interval over the 30 tasks.

Tasks `n1`-`n5` have no expected calls and no required answer, so a model that does nothing passes them.

## Results

| Model | Tasks passed | Pass rate | 95% interval | Avg tokens/s | Peak GPU memory above idle |
|---|---:|---:|---:|---:|---:|
| `qwen3:4b` | 21/30 | 70.0% | 52-83% | 13.4 | 2335 MB |
| `qwen3:8b` | 19/30 | 63.3% | 46-78% | 4.5 | 2335 MB |
| `qwen3:1.7b` | 12/30 | 40.0% | 25-58% | 78.0 | 1727 MB |
| `qwen2.5-coder:7b` | 5/30 | 16.7% | 7-34% | 5.1 | 2265 MB |

Idle GPU memory was 0 MB in the `nvidia-smi` check.

CPU/GPU placement from `ollama ps` (one snapshot per model, not per-run measurements; placement can vary):

| Model | CPU / GPU | Avg tokens/s |
|---|---|---:|
| `qwen3:1.7b` | 0% / 100% | 78.0 |
| `qwen3:4b` | 33% / 67% | 13.4 |
| `qwen2.5-coder:7b` | 55% / 45% | 5.1 |
| `qwen3:8b` | 61% / 39% | 4.5 |

### By category (individual runs, not majority-voted)

For example, `13/24` means 13 passed runs out of 24 chain runs (8 tasks x 3 runs).

| Model | Chain | No-tool | Robustness | Selection | Single |
|---|---:|---:|---:|---:|---:|
| `qwen3:4b` | 13/24 | 15/21 | 12/12 | 9/15 | 15/18 |
| `qwen3:8b` | 9/24 | 15/21 | 9/12 | 9/15 | 15/18 |
| `qwen3:1.7b` | 0/24 | 15/21 | 4/12 | 9/15 | 8/18 |
| `qwen2.5-coder:7b` | 0/24 | 15/21 | 0/12 | 0/15 | 0/18 |

## What the results show

- `qwen3:4b` scored highest (21/30). **The 4B vs 8B difference (2 tasks) is within the uncertainty at this sample size**, so this benchmark cannot say the 4B is the better model.
- `qwen3:8b` matched the 4B on single and selection runs. It was weaker on chain runs (9/24 vs 13/24), but that is a small number of tasks, so treat it as a pointer for follow-up, not a conclusion.
- Speed fell as the CPU share rose (78.0 tokens/s at 0% CPU, 13.4 at 33%, 5.1 at 55%, 4.5 at 61%). This is consistent with CPU offload slowing the larger models on a 4 GB GPU, but it rests on one placement snapshot per model. The 4B was also not fully on the GPU.
- The 4B and 8B show the same peak GPU memory (2335 MB) even though the 8B is larger, which is consistent with both being partly offloaded. This is not confirmed.
- `qwen3:1.7b` was the fastest and used the least GPU memory, and passed 12/30. It failed every chain run.
- `qwen2.5-coder:7b` passed no tool-using tasks. It wrote tool calls as plain text in 75 of 90 runs: 24 chain, 6 no-tool, 12 robustness, 15 selection and 18 single-task runs. Its 5/30 equals the never-calls baseline below, so this benchmark mostly shows a mismatch between this model's output and Ollama's structured tool calls, not its tool-use ability.
- All four models had equal no-tool counts (15/21 runs), but not the same failing tasks: `n3` failed for `qwen3:1.7b` and `qwen3:8b`; `x11` failed for all four models; `x12` failed for `qwen2.5-coder:7b` and `qwen3:4b`. On `x11`, `qwen3:1.7b` and `qwen3:8b` answered 10 instead of 9, and `qwen2.5-coder:7b` wrote an irrelevant `add` call as text. The stored `qwen3:4b` response starts with internal deliberation but is truncated at 200 characters, so the CSV does not show whether it later answered. The expected answer is unambiguous: “benchmark” has 9 letters.
- **Never-calls baseline:** a model that makes no tool calls and gives no useful answer passes 5/30 tasks (16.7%) under the current checker (`n1`-`n5`).

## Limitations

- **Small sample:** 30 tasks, written by one person. Differences of a few tasks between models are not reliable.
- **Runs are not independent:** at temperature 0 the 3 runs are nearly identical, so repeating runs adds little.
- **One machine and one runtime:** results are for Ollama on a 4 GB GPU. Other hardware or runtimes may differ.
- **Quantization is not isolated:** models are compared at their default Ollama tags, so differences mix model size, family and quantization level.
- **Placement is a snapshot:** CPU/GPU placement was recorded once per model, not per run.
- **Lenient final-answer check:** a substring match can pass a wrong answer that happens to contain the expected text.
- **No-tool tasks reward doing nothing:** `n1`-`n5` pass on an empty answer, so the no-tool category does not separate models well.
- **Fake tools:** real tool use involves messier schemas and errors.

## Related work

The following projects provide useful context for local and tool-calling evaluation:

- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) evaluates tool/function calling on real-world data and includes multi-turn and agentic evaluation. TinyToolBench is much smaller and uses a fixed set of local tools and tasks.
- [BenchLoop](https://github.com/outsourc-e/bench-loop) is a local-first benchmark CLI and web app with broader suites for quality, speed, reliability, and multi-turn agent/tool use.
- The two NeurIPS 2026 listings previously mentioned were not found in the [official 2026 listings](https://neurips.cc/Downloads/2026), so they are omitted.

**Where TinyToolBench differs:** it focuses on 30 fixed tasks using four local tools, with no LLM judge, and reports speed and hardware measurements from one constrained laptop.

## Running the benchmark

1. Install [Ollama](https://ollama.com) and pull the models:
   ```
   ollama pull qwen3:1.7b
   ollama pull qwen3:4b
   ollama pull qwen3:8b
   ollama pull qwen2.5-coder:7b
   ```
2. Activate the project virtual environment, then run:
   ```
   python run_benchmark.py
   ```
3. Raw benchmark results are written to `results/results_v3_resume.csv`. The analysis input is `results/results_v3_corrected.csv`. The raw file had 90 blank model labels because its UTF-8 BOM caused the model header to be read incorrectly; the corrected file assigns those rows to `qwen3:1.7b`. No run-level pass/fail values changed, but this restores the model attribution and its reported score of 12/30; the other models' scores are unchanged.
4. Run `python performance_table.py` for `results/performance_summary_v3.csv`, `python analyze.py` for `results/speed_v3.png` and `results/vram_v3.png`, and `python failures.py` for `results/results_v3_with_reasons.csv` and `results/failure_breakdown_v3.csv`.

## License

MIT License. See `LICENSE`.
