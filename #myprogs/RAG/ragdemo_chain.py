import os
from dotenv import load_dotenv
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from operator import itemgetter

def format_doc(docs):
    return "\n\n".join(d.page_content for d in docs)

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm=ChatOllama(base_url=url,model="qwen2.5:1.5b")
embeddings=OllamaEmbeddings(base_url=url,model="granite-embedding:30m")

document = TextLoader("C:/Users/tech/Documents/LangchainDemo/#myprogs/RAG/product-data.txt").load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=200)
chunks=text_splitter.split_documents(document)
vector_store=FAISS.from_documents(chunks,embeddings)
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","""You are an assistant for answering questions.
    Use the provided context to respond.If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.\n
    Context:\n{context}    
    """),
    ("human", "{input}")
]
)
st.title("RAG")

qa_chain = prompt_template | llm | StrOutputParser()
rag_chain = {"context": itemgetter("input") | retriever | format_doc , "input": itemgetter("input")} | qa_chain

st.write("Chat with Document")
question=st.text_input("Your Question")

if question:
    response = rag_chain.invoke({"input":question})
    st.write(response)