from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="deepseek-r1:1.5b")

ques=input("Enter ques:")
resp=llm.invoke(ques)
print(resp.content)