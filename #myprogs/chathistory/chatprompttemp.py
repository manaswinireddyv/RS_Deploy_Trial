import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
set_debug(True)

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="qwen2.5:1.5b")
prompt_temp=ChatPromptTemplate.from_messages(
    [
        ("system","You are a tenant coach. Answer any questions related to the tenant architecture and its process."),
        ("human","{input}")
    ]
)

st.title("Tenant Guide")
input=st.text_input("Enter a ques:")
chain= prompt_temp | llm
if input:
    resp=chain.invoke({"input":input})
    st.write(resp.content)