from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import base64
import streamlit as st
def encode_image(image_file):
        return base64.b64encode(image_file.read()).decode()


llm=ChatOllama(base_url="http://localhost:11434",model="llava:7b")
prompt_temp=ChatPromptTemplate(
    [
        ("system","""You are a great assistant who can analyze images of nutrition charts
          and help choose the best right diet plan."""),
        ("human",
         [
             {"type":"text","text":"{input}"},
             {
                 "type":"image_url",
                 "image_url":{
                     "url":f"data:image/jpeg;base64,""{image1}",
                     "detail":"low",
                 }
             },
             {
                 "type":"image_url",
                 "image_url":{
                     "url":f"data:image/jpeg;base64,""{image2}",
                     "detail":"low",
                 }
             }
         ]
        )
    ]
)
st.title("Diet Planner")

chain=prompt_temp | llm

upload_img1=st.file_uploader("Upload your 1st image",type=["jpg","png","jpeg"])
upload_img2=st.file_uploader("Upload your 2nd image",type=["jpg","png","jpeg"])

ques=st.text_input("Enter ur ques")
if ques and upload_img1 is not None and upload_img2 is not None:
        image1 = encode_image(upload_img1)
        image2 = encode_image(upload_img2)
        resp=chain.invoke({"input":ques,"image1":image1,"image2":image2})
        st.write(resp.content)