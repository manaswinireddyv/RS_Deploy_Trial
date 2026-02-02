from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="qwen2.5:1.5b")

accumulated = []
for chunk in llm.stream("What is the capital of India?"):
    text = chunk.content  # partial text for this chunk
    print(text, end="", flush=True)  # print as it comes
    accumulated.append(text)
print()

final_text = "".join(accumulated)
# final_text is the complete answer if you need it later
