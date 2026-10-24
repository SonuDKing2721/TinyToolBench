import ollama
from tools import TOOL_SCHEMAS

r = ollama.chat(
    model="qwen2.5-coder:7b",
    messages=[{"role": "user", "content": "What is 482 plus 19?"}],
    tools=TOOL_SCHEMAS,
    options={"temperature": 0},
)

print(r.message)