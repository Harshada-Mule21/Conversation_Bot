import os
import shutil
import warnings
import pandas as pd
import streamlit as st
import numpy as np
from streamlit_extras.switch_page_button import switch_page

warnings.warn('ignore', category=FutureWarning)
np.random.seed(1)
warnings.simplefilter("ignore")


with st.chat_message("user"):
    st.write("**Custom Speaking Bot** 👋")

UPLOAD_FOLDER_1 = 'Bot_folder/'
if not os.path.isdir(UPLOAD_FOLDER_1):
    os.mkdir(UPLOAD_FOLDER_1)
path = os.listdir('Bot_folder/')
lis = []
for i in range(len(path)):
    lis.append(path[i])
options = ["Create New Bot", "Load Existing Bot"]
selected_index = st.radio("Select an option", options, index=0)
if selected_index == "Create New Bot":
    
    a, b, c = st.columns([2, .6, 2])
    with a:
        Bot_name = st.text_input("Enter Bot_name Name")
        options_web_list = ["sitemap.xml link of website", "list of url"]
        selected_index_web_list = st.radio("Select an option", options_web_list, index=0)
        if selected_index_web_list == "sitemap.xml link of website":
            st.session_state.file_name = st.text_input("Enter the sitemap.xml link of website")
            st.session_state.sitemap = "True"
        else:
            st.session_state.web_list = st.text_input("Enter the website url")
            print(st.session_state.web_list)
            st.session_state.sitemap = "False"
        User_Submit = st.button('Create Bot', key="0")
    if User_Submit:
        if len(Bot_name.strip()) == 0:
            st.write("Bot Name Shouldn't be Empty! :sunglasses:" )
        elif Bot_name not in lis:
            st.session_state.Bot_name = Bot_name
            os.mkdir('Bot_folder/' + Bot_name + '/')
            switch_page("load_data")
        else:
            st.subheader("Bot Name Already Exist! :sunglasses:")
else:
    a, b, c = st.columns([2, .6, 2])
    with a:
    	Bot_name = st.selectbox('Select the website url to load', lis, key='1')
    
    st.session_state.Bot_name = Bot_name
    
    with a:
        User_Submit = st.button('Load Bot', key="0")
    if User_Submit:
        switch_page("speaking_bot")
