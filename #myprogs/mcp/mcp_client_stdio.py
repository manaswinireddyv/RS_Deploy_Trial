from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

import asyncio
import streamlit as st 

client= MultiServerMCPClient({
    "tools":{
        "url":"http://localhost:8000/mcp",
        "transport": "streamable_http"
    }
})


tools = asyncio.run(client.get_tools())

llm = ChatOllama(base_url="http://localhost:11434",model="llama3.1:latest")
agent = create_agent(llm, tools)

st.title("AI Agent (MCP Version)")
task = st.text_input("Assign me a task")

if task:
    response = asyncio.run(agent.ainvoke({"messages": task}))
    st.write(response)
    final_output = response["messages"][-1].content
    st.write(final_output)

