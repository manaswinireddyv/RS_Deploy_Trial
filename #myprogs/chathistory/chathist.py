import os
from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

set_debug(True)

load_dotenv()
url=os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(base_url=url,model="qwen2.5:1.5b")
prompt_temp=ChatPromptTemplate.from_messages(
    [
        ("system","You are a tenant coach. Answer any questions related to the tenant architecture and its process."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)

print("Tenant Guide")
chain= prompt_temp | llm
history_for_chain= ChatMessageHistory()
chain_with_history= RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

while True:
    ques=input("Enter a ques:")
    if ques:
        resp=chain_with_history.invoke({"input":ques},
                                        {"configurable":{"session_id":"abc123"}}
                                      )
        print(resp.content)