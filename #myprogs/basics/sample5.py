import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug
from dotenv import load_dotenv

set_debug(True)

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="qwen2.5:1.5b")


st.title("Ask something")

ques=st.text_input("Enter the question:")
if ques:
    resp=llm.invoke(ques)
    st.write(resp.content)