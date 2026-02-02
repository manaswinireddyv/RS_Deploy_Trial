import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
llm=ChatOpenAI(model="gpt-4o",api_key=OPENAI_API_KEY)

st.title("Ask ques")
ques=st.text_input("Enter the question:")
if ques:
    resp=llm.invoke(ques)
    st.write(resp.content)