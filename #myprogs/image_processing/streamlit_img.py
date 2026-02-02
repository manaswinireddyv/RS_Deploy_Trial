from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import base64
import streamlit as st
def encode_image(image_file):
        return base64.b64encode(image_file.read()).decode()


llm=ChatOllama(base_url="http://localhost:11434",model="llava:7b")
prompt_temp=ChatPromptTemplate(
    [
        ("system","""You are a great assistant who can desribe images and create 
         a very short story."""),
        ("human",
         [
             {"type":"text","text":"{input}"},
             {
                 "type":"image_url",
                 "image_url":{
                     "url":f"data:image/jpeg;base64,""{image}",
                     "detail":"low",
                 }
             }
         ]
        )
    ]
)
st.title("Image Analyzing")

chain=prompt_temp | llm

upload_img=st.file_uploader("Upload your image",type=["jpg","png","jpeg"])

ques=st.text_input("Enter ur ques")
if ques and upload_img is not None:
        image = encode_image(upload_img)
        resp=chain.invoke({"input":ques,"image":image})
        st.write(resp.content)