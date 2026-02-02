from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="gemma3:latest")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant.Explain clearly in 3 sentences."),
    ("human", "{question}")
])

chain = prompt | llm
print(chain.invoke({"question": "What is the capital of India?"}).content)