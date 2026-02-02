import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ---------------- setup ----------------
load_dotenv()
url = os.getenv("OLLAMA_BASE_URL")

mcp = FastMCP("promptandresource-mcp-demo")

# ✅ Load & index ONCE at server startup
embeddings = OllamaEmbeddings(
    base_url=url,
    model="granite-embedding:30m"
)

document = TextLoader(
    "C:/Users/tech/Documents/LangchainDemo/#myprogs/embeddings/job_listings.txt"
).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(document)

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# ---------------- MCP RESOURCE ----------------
def bharath_bio() -> str:
    """
    MCP resource that returns relevant content
    from vector store instead of hard-coded text
    """
    query = "job listings skills requirements roles"
    docs = retriever.invoke(query)

    return "\n\n".join(doc.page_content for doc in docs)


def ask_about_bharath(question: str, context: str) -> str:
    return (
        "System: You are a helpful assistant. "
        "Answer strictly using the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"User question: {question}\n"
        "Answer:"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    