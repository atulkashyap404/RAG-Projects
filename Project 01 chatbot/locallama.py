from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


## Lnagsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are ahelpful assistant. Please response tp the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain Demo with OLLama Local')
input_text=st.text_input("Search the topic u want")


## OLLama LLAma-3 model

llm=Ollama(model="llama3")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({"question":input_text}))