import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

## load the GROQ API Key and OPENAI APY key
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
groq_api_key=os.getenv('GROQ_API_KEY')

st.title("(RAG) App: LLama-3 ChatBot-🤖")


llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-70b-8192")



prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based o the provided context only.
    Please provide the most accurate response based on the question
    <cotext>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embeddins():
    
    if "vectors" not in st.session_state:
        
        
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("papers") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load()  ##Document Loading 
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)  ## chunk creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)  ## splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) ## vector store OpenAi embeddings
    



prompt1=st.text_input("Enter Your Question From Documents")


if st.button("Documents Embeddings"):
    vector_embeddins()
    st.write("Vector Store DB is Ready")
    
  
  
  
 




## creating prompt

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain) 
    
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    st.write("Response time :", time.process_time()-start)
    st.write(response['answer'])
    
    
    
    #with a streamlit expander
    with st.expander("Document Similarity Search"):
        # find the relavent chunk
        for i, doc in enumerate(response["context"]):
          st.write(doc.page_content)
          st.write("-------------------------------------------------")  
    
    