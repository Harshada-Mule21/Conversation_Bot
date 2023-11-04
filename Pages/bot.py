from streamlit_extras.switch_page_button import switch_page
# from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader,DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
import streamlit as st
import os
import pickle

OPENAI_API_KEY = "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"

embedding_file = 'Bot_folder/' + st.session_state.Bot_name + '/embedding.pkl'

file_path = 'Bot_folder/' + st.session_state.Bot_name + "/memory.pkl"
# Check if the file exists
if not os.path.exists(file_path):
    # Create the file
    empty_data = 'None'
    with open(file_path, "wb") as file:
        # Write initial content to the file if needed
        pickle.dump(empty_data,file)

def main():
    print('in main')
    # Load the OpenAI key
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    with st.chat_message("user"):
        st.write("**Custom Speaking Bot** 👋")
    
    with open(embedding_file, "rb") as f:
        vectordb = pickle.load(f)


    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    with open(file_path, "rb") as file:
        # Read the contents of the file
        contents = pickle.load(file)

    # Check if the contents of the file are empty
    if contents == 'None' or len(contents.chat_memory.dict()['messages']) == 0:
        print('in chain if')
        chain = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=0.7),verbose=True,
                                                      retriever= vectordb.as_retriever(),memory = memory)
    else:
        print('in chain else')
        print(contents)
        memory = contents
        chain = ConversationalRetrievalChain.from_llm(llm=OpenAI(temperature=0.7),verbose=True,
                                                      retriever= vectordb.as_retriever(),memory = memory)


    conversation = []
    query = st.text_input(f"Ask a question.")
    conversation.append(("User", query))

    if query is not None and query !="":
        if query.lower() in ["hi", "hello", "hey"]:
            response = "Hello! How can I assist you today?"
        else:
            result = chain({"question": query})
            response = result['answer']
        conversation.append(("ChatBot", response))
        for sender, message in conversation:
            st.write(f"{sender}: {message}")
    with open(file_path,'wb') as f:
        pickle.dump(memory,f)


if __name__=='__main__':
    main()
    
