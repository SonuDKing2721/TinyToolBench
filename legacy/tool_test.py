import ollama
from tools import TOOL_FUNCS, TOOL_SCHEMAS

messages = [{
    "role": "user",
    "content": "get my groceries from notes"
}]

reply = ollama.chat(
    model="qwen3:1.7b",
    messages=messages,
    tools=TOOL_SCHEMAS,
    options={"temperature": 0},
    think=False
)

if not reply.message.tool_calls:
    print("The model did not request a tool.")
    print(reply.message.content)
    raise SystemExit

messages.append(reply.message)

for call in reply.message.tool_calls:
    name = call.function.name
    arguments = dict(call.function.arguments)

    function = TOOL_FUNCS[name]
    result = function(**arguments)

    print(f"Python ran {name} with {arguments}: result = {result}")

    messages.append({
        "role": "tool",
        "tool_name": name,
        "content": str(result)
    })

final = ollama.chat(
    model="qwen3:1.7b",
    messages=messages,
    tools=TOOL_SCHEMAS,
    options={"temperature": 0},
    think=False
)

print("Model's final answer:", final.message.content)