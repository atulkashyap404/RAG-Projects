from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
## Lnagsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


##Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are ahelpful assistant. Please response tp the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain Demo with OPENAI API')
input_text=st.text_input("Search the topic u want")

##openAI LLM

llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))