from langchain_ollama import ChatOllama

# Start Ollama daemon first (usually automatic after install)
# Then run this Python script.

model = ChatOllama(model="llama3.1")  
for chunk in model.stream("Explain what RAG is, briefly."):
    print(chunk.content or "", end="", flush=True)
print()