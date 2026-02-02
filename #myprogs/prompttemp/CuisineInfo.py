import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
set_debug(True)

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="qwen2.5:1.5b")
prompt_template=PromptTemplate(input_variables=["country","no_of_paras","language"],
                               template="""You are an expert in traditional cuisines.
                               You provide info about a special dish from a specific country.
                               Avoid giving info about fictional places.If the country is fictional or nob-existant then answer: i dont know ,you must figure it.
                               Answer the question: What is the traditional cuisie of {country}?
                               Answer in {no_of_paras} short paras in {language}""")

st.title("Cusine Info")

country=st.text_input("Enter country:")
no_of_paras=st.number_input("Enter the no.of paras",min_value=1,max_value=3)
language=st.text_input("Enter the language")

if country:
    response = llm.invoke(prompt_template.format(country=country,
                                                 no_of_paras=no_of_paras,
                                                 language=language
                                                 ))
    st.write(response.content)



