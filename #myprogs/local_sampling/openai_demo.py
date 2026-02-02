from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Start Ollama daemon first (usually automatic after install)
# Then run this Python script.

model = ChatOllama(model="llama3.1")      # or "llama3.1:8b" if you pulled that tag
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic} in simple terms.")
])

formatted = prompt.format(topic="RAG (Retrieval-Augmented Generation)")
# LangChain expects a string prompt; format() returns that
response = model.invoke(prompt.format())
print(response.content)