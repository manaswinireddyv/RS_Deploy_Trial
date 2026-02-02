import requests

url = "http://172.168.168.232:11434/api/chat"
payload = {
  "model": "gemma3:latest",
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain RAG in one sentence."}
  ],
  "stream": False
}
print(requests.post(url, json=payload, timeout=60).json()["message"]["content"])