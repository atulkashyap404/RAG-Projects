from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv


load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

## FAST API APP INITIALIZATION

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server" 
)

## making routs

add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

model=ChatOpenAI()

## ollama llama3
llm=Ollama(model="llama3")


prompt1=ChatPromptTemplate.from_template("Write an assay for me about {topic} with 100 words") 
prompt2=ChatPromptTemplate.from_template("Write an poem for me about {topic} with 100 words") 


add_routes(
    app,
    prompt1|model,
    path="/assay"
)

add_routes(
    app,
    prompt2|llm,
    path="/poem"
)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)