import streamlit as st
import os
import openai
import pickle
import speech_recognition as sr
import pyttsx3
from langchain.embeddings.openai import OpenAIEmbeddings
# from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from gtts import gTTS
from google.cloud import speech_v1p1beta1

# Constants
embedding_file = 'Bot_folder/' + st.session_state.Bot_name + '/embedding.pkl'
#FILE_PATH = 'Bot_folder/' + st.session_state.Bot_name + "/"+ +"memory.pkl"
mp3_path = 'Bot_folder/' + st.session_state.Bot_name + "/output.mp3"
OPENAI_API_KEY = "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"

def load_memory(FILE_PATH):
    if not os.path.exists(FILE_PATH):
        with open(FILE_PATH, "wb") as file:
            pickle.dump(None, file)
    with open(FILE_PATH, "rb") as file:
        return pickle.load(file)

def save_memory(memory):
    with open(FILE_PATH, 'wb') as f:
        pickle.dump(memory, f)

def google_speech(audio):
    client = speech_v1p1beta1.SpeechClient.from_service_account_file('key.json')
    
    # Save the audio data to a file
    with open('speech.wav', 'wb') as f:
        f.write(audio.get_wav_data())
    
    # Read the audio file and set up recognition configuration
    with open('speech.wav', 'rb') as audiofile:
        config = speech_v1p1beta1.RecognitionConfig(
            language_code="fil-PH",
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True,
        )
        
        audio_content = audiofile.read()

        response = client.recognize(
            config=config,
            audio={"content": audio_content},
        )

        # Print the transcription results
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))

        return response


def recognize_speech(selected_language_input):
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=0)
    r.dynamic_energy_threshold=False
    r.energy_threshold = 150 # 300 is the default value of the SR library
    
    with mic as source:
        st.write("Speak something...")
        audio = r.listen(source)
        st.write("Done listening.")
    try:
        # query = r.recognize_google(audio, language=selected_language_input, show_all=False)
        query = google_speech(audio)
        return query
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand.")
        return None
    except sr.RequestError:
        st.write("Sorry, there was an error with the speech recognition service.")
        return None


def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'hindi')
    engine.say(text)
    engine.runAndWait()
    
def generate_response(prompt, vectordb,FILE_PATH):

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Read the contents of the history
    contents = load_memory(FILE_PATH)

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
    result = chain({"question": prompt})
    return result['answer'] , memory

def main():

    #__ = load_memory()
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    with open(embedding_file, "rb") as f:
        vectordb = pickle.load(f)

    with st.chat_message("user"):
        st.write("**Custom Speaking Bot** 👋")
    
    # Define the custom CSS style for the button
    button_style = """
    <style>
    .stButton>button {
        background-color: #4CAF50; /* Green background color */
        color: white; /* White text color */
    .stButton>button:hover {
        background-color: #45a049; /* Darker green color on hover */
    }
    style>
    """

    # Display the custom CSS style
    st.markdown(button_style, unsafe_allow_html=True)
    st.write("Click the SPEAK and speak your question:")
    
    language_mapping = {
        'English': 'en',
        'Filipino': 'tl',
        'Hindi' : 'hi',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
        'Italian': 'it',
        'Japanese': 'ja',
        'Chinese': 'zh',}
    # Create a SelectBox to display language names

    language_mapping_input = {
        'English': 'en-US',
        'Filipino': 'fil-PH',
        'Hindi' : 'hi-IN',
        'Spanish': 'es-ES',
        'French': 'fr-FR',
        'German': 'de-DE',
        'Italian': 'it-IT',
        'Japanese': 'ja-JP',
        'Chinese': 'zh-CN',}
    
    col1, col2 = st.columns(2)
    with col1:
        selected_language_name = st.selectbox('Select a Speech language for I/P', list(language_mapping.keys()), key='3')
        selected_language_input = language_mapping.get(selected_language_name)

    with col2:
        selected_language_name_out = st.selectbox('Select a Speech language for O/P', list(language_mapping.keys()), key='4')
        selected_language_code = language_mapping.get(selected_language_name_out)
    
    FILE_PATH = 'Bot_folder/' + st.session_state.Bot_name + "/"+ selected_language_name +"_memory.pkl"
    
    # Check if the file exists
    if not os.path.exists(FILE_PATH):
        # Create the file
        empty_data = 'None'
        with open(FILE_PATH, "wb") as file:
            # Write initial content to the file if needed
            pickle.dump(empty_data,file)
      
    conversation = []
    query = "None"
    but = st.button("SPEAK")
  
    # output = "Welcome to Speakingbot"
    if but:
        query = recognize_speech(selected_language_input)
        if query is not None and query.strip():
            st.markdown(f"**Query:**")
            st.write(f"{query}")

    if query != "None":
        # selected_language_name = st.selectbox('Select a Speech language', list(language_mapping.keys()), key='3')
        # selected_language_code = language_mapping.get(selected_language_name)
        conversation.append(("User", query))
        if query.lower() in ["hi", "hello", "hey", "hi.", "hello.", "hey."]:
            output = "Hello! How can I help you?"
        else:
            output, memory_pass = generate_response(query, vectordb, FILE_PATH)
            with open(FILE_PATH,'wb') as f:
                pickle.dump(memory_pass,f)

        conversation.append(("ChatBot", output))
            
        st.markdown(f"**Output:**")
        st.write(f"{output}")
            #text_to_speech(output)
        
        tts = gTTS(text=output, lang=selected_language_code, slow=False)
            # Save the generated speech as an audio file (you can also use a temporary file)
        tts.save(mp3_path)
        with open(mp3_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

            
            
                
if __name__ == '__main__':
    main()

