import asyncio
from langchain_ollama import ChatOllama
async def main():
    model = ChatOllama(model="llama3.1")
    resp = await model.ainvoke("What’s the difference between LangChain and LlamaIndex?")
    print(resp.content)

asyncio.run(main())