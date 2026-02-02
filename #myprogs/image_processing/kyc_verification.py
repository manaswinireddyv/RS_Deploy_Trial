from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import base64
import streamlit as st
from datetime import date
def encode_image(image_file):
        return base64.b64encode(image_file.read()).decode()

llm=ChatOllama(base_url="http://localhost:11434",model="llava:7b")
prompt_temp=ChatPromptTemplate(
    [
        ("system","""You are a helpful assistant that can verify identification documents."""),
        ("human",
         [
             {"type":"text","text":"Verify the identification details"},
             {"type":"text","text":"{name}"},
             {"type":"text","text":"{dob}"},
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
st.title("KYC Verification")

chain=prompt_temp | llm

upload_img=st.file_uploader("Upload your image",type=["jpg","png","jpeg"])


name=st.text_input("Enter ur name")
dob=st.date_input("Enter ur dob",min_value=date(1900,1,1))

if dob and name and upload_img is not None:
    image = encode_image(upload_img)
    dob_str = dob.isoformat()
    resp=chain.invoke({"name":name,"dob":dob,"image":image})
    st.write(resp.content)