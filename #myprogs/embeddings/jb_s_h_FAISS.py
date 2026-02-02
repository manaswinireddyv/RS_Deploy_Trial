import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm=OllamaEmbeddings(base_url=url,model="granite-embedding:30m")

document = TextLoader("C:/Users/tech/Documents/LangchainDemo/#myprogs/embeddings/job_listings.txt").load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=200,
                                              chunk_overlap=10)
chunks=text_splitter.split_documents(document)
db=FAISS.from_documents(chunks,llm)
retriever = db.as_retriever()

text = input("Enter the query")

docs = retriever.invoke(text)

print(docs)
print(docs[0].page_content+"1st result")
for doc in docs:
    print(doc.page_content)