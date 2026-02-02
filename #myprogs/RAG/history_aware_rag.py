import os
from dotenv import load_dotenv
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from  langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm=ChatOllama(base_url=url,model="qwen2.5:1.5b")
embeddings=OllamaEmbeddings(base_url=url,model="granite-embedding:30m")

document = TextLoader("C:/Users/tech/Documents/LangchainDemo/#myprogs/RAG/product-data.txt").load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=200)
chunks=text_splitter.split_documents(document)
vector_store=Chroma.from_documents(chunks,embeddings)
retriever = vector_store.as_retriever()

st.title("RAG-History Aware")
# -----------------------------
# 1️⃣ Prompt for history-aware retriever
#    (NO {context} HERE)
# -----------------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that reformulates follow-up
questions into standalone questions.

Use the chat history and the latest user input to create
a self-contained question.

Do NOT answer the question, only rewrite it."""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

qa_prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","""You are an assistant for answering questions.
    Use the provided context to respond.If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    context:{context}  """),
        MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
]
)

qa_chain = create_stuff_documents_chain(llm=llm,prompt=qa_prompt_template)
rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)


history_for_chain=StreamlitChatMessageHistory()
chain_for_history=RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

st.write("Chat with Document")
question=st.text_input("Your Question")

if question:
    response = chain_for_history.invoke({"input":question},{"configurable":{"session_id":"abc123"}})
    st.write(response['answer'])