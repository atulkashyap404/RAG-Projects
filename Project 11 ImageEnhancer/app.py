from dotenv import load_dotenv
import os
import time
from openai import OpenAI
import streamlit as st
import requests
from io import BytesIO


# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
api_key=os.getenv('OPENAI_API_KEY')



# Pass the API key to the OpenAI client
client = OpenAI(api_key=api_key)






def imageGen(input):
  response = client.images.generate(
    model=model,
    prompt=input,
    size=size,
    quality="hd",
    n=int(number),
  )
  img_url=response.data[0].url
  return img_url

st.set_page_config(
    page_title="Image-Gen",
    page_icon="🐼",
)

st.title('Image-Gen 🐼')

input=st.sidebar.text_area("Image Prompt")
model=st.sidebar.selectbox(
  "Select model",
  ("dall-e-2", "dall-e-3")
)
size=st.sidebar.selectbox(
  "Image size",
  ("1024x1024", "1024x1792", "1792x1024")
)
number=st.sidebar.selectbox(
  "Select Number of Image",
  ("1")
)




if st.button("Generate Image"):
    with st.spinner('Wait for it...'):
        img_url = imageGen(input)
        st.image(img_url)
        time.sleep(5)
        

        # Fetch the image content from the URL
        img_response = requests.get(img_url)
        img_bytes = BytesIO(img_response.content)
        
        # Add the download button
        st.download_button(
            label="Download Image",
            data=img_bytes,
            file_name="generated_image.png",
            mime="image/png",
        )
        st.success('Done!')
    

    
  


