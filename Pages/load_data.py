import os
import shutil
import warnings
import pandas as pd
import streamlit as st
import numpy as np
from streamlit_extras.switch_page_button import switch_page
import requests
from bs4 import BeautifulSoup
# from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import ast

OPENAI_API_KEY = "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"

warnings.warn('ignore', category=FutureWarning)
np.random.seed(1)
warnings.simplefilter("ignore")

ALLOWED_EXTENSIONS = (['xml'])

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re

# Function to check if a list item contains a URL
def contains_url(item):
    url_pattern = r'https?://\S+'
    return re.search(url_pattern, item) is not None

# Function to get all the unique URLs from a webpage
def get_unique_urls(base_url):
    # Initialize a set to store the unique URLs
    unique_urls = set()
    
    unique_urls.add(base_url)
    # Send an HTTP GET request to the base_url
    response = requests.get(base_url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all anchor tags (a) with href attributes
        for anchor in soup.find_all('a', href=True):
            # Get the href attribute value
            href = anchor['href']
            
            # Join the URL with the base_url to handle relative URLs
            full_url = urljoin(base_url, href)
            
            # Add the full URL to the set (duplicates will be automatically removed)
            unique_urls.add(full_url)
    
    # Filter the list to get URLs
    url_list = [item for item in unique_urls if contains_url(item)]
    # Convert the set back to a list
    return list(url_list)
    

def parse_sitemap(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    urls = [element.text for element in soup.find_all("loc")]
    return urls
    
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
def find_vector_emb(sites, Bot_folder):

    print("save file: ",  str(Bot_folder + "embedding.pkl"))
    sites_filtered = [url for url in sites if '/reference/' not in url and '?hl' not in url]
    print(len(sites_filtered))
    
    loaders = UnstructuredURLLoader(urls=sites_filtered)
    data = loaders.load()
    
    text_splitter = CharacterTextSplitter(separator='\n', 
                                      chunk_size=1000, 
                                      chunk_overlap=200)
                              
    docs = text_splitter.split_documents(data)
    
    embeddings = OpenAIEmbeddings()
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)
    
    #save embedding 
    with open(str(Bot_folder + "embedding.pkl"), "wb") as f:
        pickle.dump(vectorStore_openAI, f)
    


with st.chat_message("user"):
    st.write("**Custom Speaking Bot** 👋")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


if st.session_state.sitemap == "True":
    file = st.session_state.file_name
    if file:
        UPLOAD_FOLDER = 'Bot_folder/' + st.session_state.Bot_name + '/' 
    
        if file and allowed_file(file):
            print("*******************************", file)
            if not os.path.isdir(UPLOAD_FOLDER):
                os.mkdir(UPLOAD_FOLDER)
            st.session_state.file_name = file
            #with st.spinner('Wait for it...'):
            sites = parse_sitemap(file)
            find_vector_emb(sites, UPLOAD_FOLDER)
            st.success('Done!')

            switch_page("speaking_bot")
        else:
            st.subheader(':red[Unsupported xml Format] :sunglasses:')
else:
    
    input_str = st.session_state.web_list
    web_list = get_unique_urls(input_str)
    print(web_list, type(web_list))
    
    if isinstance(web_list, list):
        UPLOAD_FOLDER = 'Bot_folder/' + st.session_state.Bot_name + '/' 

        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)

        #with st.spinner('Wait for it...'):
        find_vector_emb(web_list, UPLOAD_FOLDER)
        st.success('Done!')

        switch_page("speaking_bot")
    else:
        st.subheader(':red[Incorrect list Format] :sunglasses:')
