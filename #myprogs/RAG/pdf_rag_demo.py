import os
from dotenv import load_dotenv
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from  langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm=ChatOllama(base_url=url,model="qwen2.5:1.5b")
embeddings=OllamaEmbeddings(base_url=url,model="granite-embedding:30m")

document = PyPDFLoader("C:/Users/tech/Documents/LangchainDemo/#myprogs/RAG/academic_research_data.pdf").load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=200)
chunks=text_splitter.split_documents(document)
vector_store=Chroma.from_documents(chunks,embeddings)
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","""You are an assistant for answering questions.
    Use the provided context to respond.If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
    
    """),
    ("human", "{input}")
]
)

qa_chain = create_stuff_documents_chain(llm=llm,prompt=prompt_template)
rag_chain = create_retrieval_chain(retriever,qa_chain)

print("Chat with Document")
question=input("Your Question")

if question:
    response = rag_chain.invoke({"input":question})
    print(response['answer'])