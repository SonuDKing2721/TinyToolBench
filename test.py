import ollama
r = ollama.chat(model="qwen3:1.7b", messages=[{"role": "user", "content": "Say hi in one sentence."}])
print(r["message"]["content"])
