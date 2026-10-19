import csv
import json
import os
import subprocess
import sys
import time

import ollama
from tools import TOOL_FUNCS, TOOL_SCHEMAS

MODELS = ["qwen3:1.7b", "qwen3:4b", "qwen2.5-coder:7b", "qwen3:8b"]
RUNS = 3
MAX_TURNS = 5

SOURCE_FILE = "results/results_v2_resume.csv"
OUT_FILE = "results/results_v3_resume.csv"

FIELDS = [
    "model",
    "task_id",
    "type",
    "run",
    "passed",
    "n_calls",
    "elapsed_s",
    "tokens_per_s",
    "vram_mb",
    "calls",
    "final_text",
    "status",
    "error",
]


def vram_used_mb():
    """Return the GPU memory currently in use, in MB."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def chat_kwargs(model):
    """Disable Qwen3 thinking so model comparisons are fairer."""
    if model.startswith("qwen3"):
        return {"think": False}
    return {}


def run_task(model, task):
    messages = [{"role": "user", "content": task["prompt"]}]
    calls_made = []
    eval_count = 0
    eval_ns = 0
    final_text = ""
    start = time.time()

    for _ in range(MAX_TURNS):
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            options={"temperature": 0},
            **chat_kwargs(model),
        )

        eval_count += response.eval_count or 0
        eval_ns += response.eval_duration or 0
        message = response.message

        if not message.tool_calls:
            final_text = message.content or ""
            break

        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = dict(tool_call.function.arguments)

            calls_made.append({
                "name": name,
                "args": arguments,
            })

            function = TOOL_FUNCS.get(name)

            try:
                if function is None:
                    result = f"error: unknown tool {name}"
                else:
                    result = function(**arguments)
            except Exception as error:
                result = f"error: {error}"

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": str(result),
            })
    else:
        final_text = f"ERROR: reached the {MAX_TURNS}-turn limit"

    elapsed = time.time() - start
    tokens_per_second = (
        eval_count / (eval_ns / 1e9)
        if eval_ns
        else 0
    )

    return calls_made, final_text, elapsed, tokens_per_second


def normalize(call):
    """Make calls comparable while preserving argument names."""
    name = call["name"]
    arguments = call["args"]

    if name in ("add", "multiply"):
        keys = ",".join(sorted(arguments.keys()))
        values = ",".join(
            sorted(str(value).lower() for value in arguments.values())
        )
        return f"{name}:{keys}:{values}"

    return f"{name}:{json.dumps(arguments, sort_keys=True)}"


def score(task, calls_made, final_text):
    expected_calls = task["expected_calls"]

    actual = sorted(normalize(call) for call in calls_made)
    expected = sorted(normalize(call) for call in expected_calls)

    if actual != expected:
        return False

    if "expected_answer" in task:
        expected_answer = str(task["expected_answer"])
        answer_to_check = final_text.replace(",", "")

        if expected_answer not in answer_to_check:
            return False

    return True


def row_key(row):
    """Identify one model/task/run combination."""
    return (
        str(row.get("model", "")),
        str(row.get("task_id", "")),
        str(row.get("run", "")),
    )


def create_output_from_old_results():
    """Copy old results into the new CSV and add status columns."""
    previous_rows = []

    if os.path.exists(SOURCE_FILE):
        with open(SOURCE_FILE, newline="", encoding="utf-8") as old_file:
            previous_rows = list(csv.DictReader(old_file))

    existing_runs = set()

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as new_file:
        writer = csv.DictWriter(new_file, fieldnames=FIELDS)
        writer.writeheader()

        for old_row in previous_rows:
            key = row_key(old_row)

            if key in existing_runs:
                continue

            old_text = old_row.get("final_text", "")
            old_status = "error" if old_text.startswith("ERROR:") else "ok"

            migrated_row = {
                field: old_row.get(field, "")
                for field in FIELDS
            }
            migrated_row["status"] = old_status
            migrated_row["error"] = (
                old_text if old_status == "error" else ""
            )

            writer.writerow(migrated_row)
            existing_runs.add(key)

    print(f"Copied {len(existing_runs)} old rows into {OUT_FILE}.")
    return existing_runs


def main():
    with open("tasks.json", encoding="utf-8") as file:
        tasks = json.load(file)

    models = sys.argv[1:] or MODELS
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(OUT_FILE) or os.path.getsize(OUT_FILE) == 0:
        existing_runs = create_output_from_old_results()
    else:
        with open(OUT_FILE, newline="", encoding="utf-8") as file:
            saved_rows = list(csv.DictReader(file))

        if not saved_rows:
            existing_runs = create_output_from_old_results()
        else:
            existing_runs = {row_key(row) for row in saved_rows}
            print(f"Found {len(existing_runs)} saved rows in {OUT_FILE}.")

    with open(OUT_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        for model in models:
            for task in tasks:
                for run_number in range(1, RUNS + 1):
                    key = (model, str(task["id"]), str(run_number))

                    if key in existing_runs:
                        print(model, task["id"], run_number, "SKIP")
                        continue

                    run_status = "ok"
                    error_message = ""

                    try:
                        calls, final_text, seconds, tokens_per_second = run_task(
                            model, task
                        )
                        passed = score(task, calls, final_text)

                        if final_text.startswith("ERROR:"):
                            run_status = "error"
                            error_message = final_text

                    except Exception as error:
                        calls = []
                        final_text = f"ERROR: {error}"
                        seconds = 0
                        tokens_per_second = 0
                        passed = False
                        error_message = str(error)

                        error_text = error_message.lower()
                        if "timeout" in error_text or "timed out" in error_text:
                            run_status = "timeout"
                        else:
                            run_status = "error"

                    try:
                        vram = vram_used_mb()
                    except Exception:
                        vram = ""

                    writer.writerow({
                        "model": model,
                        "task_id": task["id"],
                        "type": task["type"],
                        "run": run_number,
                        "passed": passed,
                        "n_calls": len(calls),
                        "elapsed_s": round(seconds, 2),
                        "tokens_per_s": round(tokens_per_second, 1),
                        "vram_mb": vram,
                        "calls": json.dumps(calls),
                        "final_text": final_text[:200],
                        "status": run_status,
                        "error": error_message,
                    })
                    file.flush()
                    existing_runs.add(key)

                    result = "PASS" if passed else "FAIL"
                    print(
                        model,
                        task["id"],
                        run_number,
                        result,
                        f"({run_status})",
                    )


if __name__ == "__main__":
    main()