from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug

set_debug(True)

llm = ChatOllama(base_url="http://localhost:11434",model="llama3.1:latest")

ques=input("Enter the question: ")
resp=llm.invoke(ques)
print(resp.content)