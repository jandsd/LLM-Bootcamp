from streamlit_chat import message
import streamlit as st
from langchain.vectorstores import FAISS
from langchain import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from io import StringIO
from PIL import Image
# Load environment variables from .env file
load_dotenv()
st.set_page_config( 
                   page_title="Youtube Bot", 
                   page_icon = Image.open('assets/dsd_icon.png'))
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.image('assets/dsdojo.webp')

OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type="password")
 
if OPENAI_API_KEY:
    # OpenAI API key
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    # Create the embeddings object
    embeddings = OpenAIEmbeddings()


    # *********************** Utils ***********************
    def create_db_from_youtube_video_url(video_urls):
        # Create an empty list to store the transcripts and documents
        transcripts = []
        documents = []
        # Load the transcript from each video URL and append it to the transcripts list
        for url in video_urls:
            loader = YoutubeLoader.from_youtube_url(url)
            transcript = loader.load()
            transcripts.append(transcript)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        # Appending each video transcript to document list
        for transcript in transcripts:
            documents.append(text_splitter.split_documents(transcript))
        documents = [item for sublist in documents for item in sublist]
            # Create the vector database
        db = FAISS.from_documents(documents, embeddings)
        return db



    # Get the answer to the question
    def get_response_from_query(db, query):
        # Search the vector database for the most similar chunks
        documents = db.similarity_search(query, k=4)

        # Get the text of the most similar chunks and concatenate them
        content = " ".join([d.page_content for d in documents])

        # Get the large language model (gpt-3.5-turbo)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)

        # Create the prompt template
        prompt_template = """
            You are a helpful assistant that that can answer questions about youtube videos 
            based on the video's transcript: {documents}
    
            Only use the factual information from the transcript to answer the question.
    
            If you feel like you don't have enough information to answer the question, say "I don't know".
    
            Always when answering, dont mention the word "transcript" say "video" instead.
    
            Your answers should be verbose and detailed
            """

        system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)

        user_template = "Answer the following question: {question}"
        user_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

        # Create the chat prompt (the prompt that will be sent to the language model)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])

        # Create the chain (that will send the prompt to the language model and return the response)
        chain = LLMChain(llm=llm, prompt=chat_prompt)

        # Get the response from the chain
        response = chain.run(question=query, documents=content)
        response = response.replace("\n", "")

        return response


    def generate_response(query, urls):
        # Create the vector database
        db = create_db_from_youtube_video_url(urls)
        response = get_response_from_query(db, query)
        return response


    # *********************** Streamlit App ***********************

    try:

        # Create a button to toggle between "Files" and "Manual" input
        input_option = st.radio("Choose input method:", ("Files", "Manual"))
        links = []
        if 'links' not in st.session_state:
            st.session_state['links'] = links


        if input_option == "Files":
            st.header("File Upload")
            # Upload a text file containing the links of the YouTube videos
            uploaded_file = st.file_uploader("Upload a text file containing the links of the YouTube videos", type="txt")

            # print(uploaded_file)
            if uploaded_file is not None:
                # Convert the uploaded file to a string
                stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
                # Read the links from the string
                links = [line.strip() for line in stringio.readlines()]

                st.session_state['links'].extend(link for link in links if link not in st.session_state['links'] and link != '')

        else:
            st.header("Manual Input")
            # Get the links of the YouTube videos from the user
            links = st.text_area("Enter the links of the YouTube videos (one link per line)", height=200).splitlines()
            st.session_state['links'].extend(link for link in links if link not in st.session_state['links'] and link != '')

        reply_container = st.container()
        container = st.container()

        # Storing the chat
        if 'question' not in st.session_state:
            st.session_state['question'] = []

        if 'answer' not in st.session_state:
            st.session_state['answer'] = []

        # Get the question from the user
        with container:
            question = st.text_input("Question:", placeholder="Ask about your Documents", key='input')

            if question:
                res = generate_response(question, st.session_state['links'])
                st.session_state['question'].append(question)
                st.session_state['answer'].append(res)

        if st.session_state['answer']:
            with reply_container:
                for i in range(len(st.session_state['answer'])):
                    user_message_key = str(i) + '_user'
                    answer_message_key = str(i) + '_answer'
                    
                    message(st.session_state['question'][i], is_user=True, key=user_message_key)
                    message(st.session_state["answer"][i], key=answer_message_key)

    except Exception as e:
        st.error(e)