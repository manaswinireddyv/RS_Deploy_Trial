from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

llm=ChatOllama(base_url="http://localhost:11434",model="llava:7b")
image = encode_image("C:/Users/tech/Documents/LangchainDemo/#myprogs/image_processing/airport_terminal_journey.jpeg")
prompt_temp=ChatPromptTemplate(
    [
        ("system","""You are a great assistant who can desribe images and create a very short story."""),
        ("human",
         [
             {"type":"text","text":"{input}"},
             {
                 "type":"image_url",
                 "image_url":{
                     "url":f"data:image/jpeg;base64,{image}",
                     "detail":"low",
                 }
             }
         ]
        )
    ]
)


chain=prompt_temp | llm
resp=chain.invoke({"input":"Explain"})
print(resp.content)