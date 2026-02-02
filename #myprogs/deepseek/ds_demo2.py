import re
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="deepseek-r1:1.5b")
embedder=OllamaEmbeddings(base_url=url,model="deepseek-r1:1.5b")

docs=TextLoader("C:/Users/tech/Documents/LangchainDemo/#myprogs/deepseek/product-data.txt").load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=text_splitter.split_documents(docs)
db=Chroma.from_documents(chunks,embedder)
retriever=db.as_retriever()

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","""You are a good assistant ,you can 
         answer any question.Limit to 2 to 3 sentences.
         {context}"""),
        ("human","{input}")
    ]
)


c1=create_stuff_documents_chain(llm,prompt)
c2=create_retrieval_chain(retriever,c1)


ques=input("Enter ques:")
if ques:
    doc=c2.invoke({"input":ques})
    final_response=re.sub(r'<think>.*?</think>','',doc['answer'],flags=re.DOTALL).strip()
    print(final_response)



