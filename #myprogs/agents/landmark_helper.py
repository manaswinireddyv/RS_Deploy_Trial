from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import streamlit as st
from langchain_core.globals import set_debug
import base64
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate
# ------------------------------
# 1. Helper to encode image
# ------------------------------
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode()

set_debug((True))
# ------------------------------
# 2. LLM setup
# ------------------------------
llm=ChatOllama(base_url="http://localhost:11434",model="qwen3-vl:latest")
# ------------------------------
# 3. Vision prompt (identify landmark)
# ------------------------------

prompt_temp=ChatPromptTemplate(
    [
        ("system","""You are a great agent which can analyze images and identify the landmarks."""),
        ("human",
         [
             {"type":"text","text":"Return the landmark name."},
            {
                    "type": "image_url",
                    "image_url": {
                        # NOTE: `image` will be passed at runtime via chain.invoke({"image": ...})
                        "url": "data:image/jpeg;base64,{image}",
                        "detail": "low",
                    },
                },
         ])
    ]
)

vision_chain= prompt_temp | llm

# ------------------------------
# 4. Tools (Wikipedia + DuckDuckGo)
# ------------------------------

tools=load_tools(["wikipedia","ddg-search"])

# ------------------------------
# 5. ReAct-style system prompt
# ------------------------------

react_system_prompt="""
You are a ReAct-style AI agent.

Follow this loop carefully:
1. THOUGHT: Think step by step about what to do next.
2. ACTION: When needed, call one of the tools (wikipedia, ddg-search).
3. OBSERVATION: Read the tool result and decide the next step.

Repeat THOUGHT → ACTION → OBSERVATION
until you are ready to give the final answer.

When you are confident, stop using tools and respond with a clear, concise final answer to the user.
"""


# ------------------------------
# 6. Create Agent (new v1 API)
# ------------------------------

#TODO: Create Agent
agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt=react_system_prompt
    )

# ------------------------------
# 7. Streamlit UI
# ------------------------------

st.title("Landmark Helper (Vision + ReAct Agent)")

uploaded_file = st.file_uploader("Upload your image", type=["jpg", "png"])
question = st.text_input("Enter a question about the landmark")

task=None

# First: use vision chain to get landmark name
if uploaded_file and question:
    image_b64 = encode_image(uploaded_file)
    vision_response = vision_chain.invoke({"image": image_b64})
    landmark_name = vision_response.content
    task = question + " " + landmark_name

if task:
    result=agent.invoke(
        {
            "messages":[{"role":"user","content":task + " without explanation"}]
        }
    )
    final_msg=result["messages"][-1]
    st.write(final_msg.content)