import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
set_debug(True)

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="qwen2.5:1.5b")
prompt_template=PromptTemplate(
    input_variables=["city","month","no_of_days","language","budget"],
    template="""Welcome to the {city} travel guide!!
    If you're visiting in {month} for {no_of_days} days, here's what you can do:
    1. Must-visit attractions you can visit within those days you are going to visit.
    2. Local cuisines you must try.
    3. Useful phrases in {language}.
    4. Budget you might need.
    5. Tips for traveling on a {budget} budget.
    Enjoy the trip!!!""" ) 

st.title("Travel Guide")

city = st.text_input("Enter the city:")
month = st.text_input("Enter the month of travel")
no_of_days=st.number_input("Enter no_of_days:",min_value=1,max_value=30)
language=st.text_input("Enter language:")
budget=st.selectbox("Choose travel budget:",["Low","Medium","High"])

chain = prompt_template | llm

if city and month and language and budget:
    response = chain.invoke({"city":city,
                             "month":month,
                             "no_of_days":no_of_days,
                             "language":language,
                             "budget":budget
                             })
    st.write(response.content)