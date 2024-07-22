import requests
import streamlit as st

def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/assay/invoke", json={'input':{'topic':input_text}})
    return response.json()['output']['content']


def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke", json={'input':{'topic':input_text}})
    return response.json()['output']



## streamlit framework

st.title('Langchain Demo with LLama-3 & OpneAI API')
input_text1=st.text_input("Write an assay on") 
input_text2=st.text_input("Write an poem on") 


if input_text1:
    st.write(get_openai_response(input_text1))

if input_text2:
    st.write(get_ollama_response(input_text2))