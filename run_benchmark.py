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
OUT_FILE = "results/results_v2_resume.csv"

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

        # Keep the model's tool request in the conversation history.
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

            # Give the tool result back to the model.
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
        # Their math is unchanged if the two values are swapped.
        keys = ",".join(sorted(arguments.keys()))
        values = ",".join(
            sorted(str(value).lower() for value in arguments.values())
        )
        return f"{name}:{keys}:{values}"

    # For lookup tools, both the argument names and values must match.
    return f"{name}:{json.dumps(arguments, sort_keys=True)}"


def score(task, calls_made, final_text):
    expected_calls = task["expected_calls"]

    actual = sorted(normalize(call) for call in calls_made)
    expected = sorted(normalize(call) for call in expected_calls)

    if actual != expected:
        return False

    if "expected_answer" in task:
        expected_answer = task["expected_answer"]
        answer_to_check = final_text.replace(",", "")

        if expected_answer not in answer_to_check:
            return False

    return True


def main():
    with open("tasks.json", encoding="utf-8") as file:
        tasks = json.load(file)

    # A model name after the command runs only that model.
    # With no model name, the MODELS list above is used.
    models = sys.argv[1:] or MODELS

    os.makedirs("results", exist_ok=True)
    new_file = (
        not os.path.exists(OUT_FILE)
        or os.path.getsize(OUT_FILE) == 0
    )

    with open(OUT_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if new_file:
            writer.writeheader()

        for model in models:
            for task in tasks:
                for run_number in range(1, RUNS + 1):
                    try:
                        calls, final_text, seconds, tokens_per_second = run_task(
                            model, task
                        )
                        passed = score(task, calls, final_text)
                    except Exception as error:
                        calls = []
                        final_text = f"ERROR: {error}"
                        seconds = 0
                        tokens_per_second = 0
                        passed = False

                    writer.writerow({
                        "model": model,
                        "task_id": task["id"],
                        "type": task["type"],
                        "run": run_number,
                        "passed": passed,
                        "n_calls": len(calls),
                        "elapsed_s": round(seconds, 2),
                        "tokens_per_s": round(tokens_per_second, 1),
                        "vram_mb": vram_used_mb(),
                        "calls": json.dumps(calls),
                        "final_text": final_text[:200],
                    })
                    file.flush()

                    status = "PASS" if passed else "FAIL"
                    print(model, task["id"], run_number, status)


if __name__ == "__main__":
    main()