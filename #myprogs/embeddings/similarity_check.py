import os
from langchain_core.globals import set_debug
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
import numpy as np

set_debug(True)
load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = OllamaEmbeddings(base_url=url,model="granite-embedding:30m")

t1=input("Enter text1:")
t2=input("Enter text2:")
r1=llm.embed_query(t1)
r2=llm.embed_query(t2)

similarity_score=np.dot(r1,r2)
print(similarity_score*100,'%')