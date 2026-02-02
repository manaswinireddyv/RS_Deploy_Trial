import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import streamlit as st
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv()
set_debug((True))

# ------------------------------
# 1. LLM setup
# ------------------------------

# ------------------------------
# 2. Tools (Wikipedia + DuckDuckGo)
# ------------------------------

tools=load_tools(["wikipedia","ddg-search"])

url=os.getenv("OLLAMA_BASE_URL")
llm=ChatOllama(base_url="http://localhost:11434",model="llama3.1:latest",
 format="json"  # nudge toward structured output (not guaranteed)
).bind_tools(tools)



# ------------------------------
# 3. ReAct-style system prompt
# ------------------------------

react_system_prompt="""
You are a ReAct-style AI agent.
And you are a tool-calling agent.
Available tools:
{tools}

Follow this loop carefully:
1. THOUGHT: Think step by step about what to do next.
2. ACTION: When needed, call one of the tools (wikipedia, ddg-search) compulsory, do not generate answers on your knowledge .Try to call any one of the tool and search in it 
and accordingly get the knowledge on it and generate the final answer.Also mention the tool you are using in the tool calls must and should.
3. OBSERVATION: Read the tool result and decide the next step.

Repeat THOUGHT → ACTION → OBSERVATION
until you are ready to give the final answer.

When you are confident, stop using tools and respond with a clear, concise final answer of the user's question to the user.Also return the tool call you made and tools used and process.
"""


# ------------------------------
# 4. Create Agent (new v1 API)
# ------------------------------

#TODO: Create Agent
agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt
    )

# ------------------------------
# 5. Streamlit UI
# ------------------------------

st.title("AI Agent (ReAct style – LangChain v1)")
task=st.text_input("Enter a task")
if task:
    result=agent.invoke(
        {
            "messages":[{"role":"user","content":task}]
        }
    )
    st.write(result)
    final_msg=result["messages"][-1]
    st.write(final_msg.content)