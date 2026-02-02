import os
from langchain_core.globals import set_debug
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings

set_debug(True)
load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = OllamaEmbeddings(base_url=url,model="granite-embedding:30m")

text=input("Enter the text:")
resp=llm.embed_query(text)
print(resp)
print(len(resp)) # vector dimension
print(resp[:10])  # preview first 10 numbers