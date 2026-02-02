import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# load env and model
load_dotenv()
url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(base_url=url, model="qwen2.5:1.5b")

# per-session id (one per Streamlit session/tab)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
session_id = st.session_state["session_id"]

# prompt with history slot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a tenant coach. Answer questions about tenant architecture and its process."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

# Streamlit-backed memory (stores messages in st.session_state)
history = StreamlitChatMessageHistory(key="chat_history")

# wire history into the chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda sid: history,                 # return the memory object for this sid
    input_messages_key="input",
    history_messages_key="chat_history",
)

st.title("Tenant Guide")

user_input = st.text_input("Enter a question:")

if user_input:
    resp = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    st.write(resp.content)

col1, col2 = st.columns(2)
if col1.button("New chat"):
    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state.pop("chat_history", None)  # clear messages for UI session
    st.rerun()

st.write("HISTORY (object):")
st.write(history)