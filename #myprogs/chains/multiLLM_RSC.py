import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import PromptTemplate
set_debug(True)

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm1 = ChatOllama(base_url=url,model="smollm2:135m")
title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer.
    You need to craft an impactful title for a speech 
    on the following topic: {topic}
    Answer exactly with one title.	
    """
)

llm2 = ChatOllama(base_url=url,model="qwen2.5:1.5b")
speech_prompt = PromptTemplate(
    input_variables=["title", "emotion"],
    template="""You need to write a powerful {emotion} speech of 350 words
     for the following title: {title}   
     Format the output with 2 keys:'title','speech' and fill them with respective values.
     Return ONLY those valid JSON with their keys "title" and "speech" as asked below.
     Do NOT include code fences, backticks, or any extra text.
    """
)

first_chain=title_prompt | llm1 | StrOutputParser() | (lambda title:(st.write(title),title)[1])
second_chain=speech_prompt | llm2 | JsonOutputParser()
final_chain= first_chain | (lambda title:{"title":title,"emotion":emotion}) | second_chain

st.title("Speech Generator")

topic = st.text_input("Enter the topic:")
emotion = st.text_input("Enter the emotion:")

if topic and emotion:
    response = final_chain.invoke({"topic":topic})
    st.write(response)
    st.write(response["title"])