import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Ensure the OpenAI API key is set
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

st.title("Question Answering App Using LangChain")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents.extend(loader.load())

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create the vectorstore to use as the index
    db = Chroma.from_documents(texts, embeddings, persist_directory="/tmp/chroma")

    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Create a chain to answer questions
    qa = ConversationalRetrievalChain.from_llm(OpenAI(api_key=openai_api_key), retriever)

    # Initialize chat history in the session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display the chat history
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.write("**Question:**", chat["question"])
            st.write("**Answer:**", chat["answer"])
            st.write("---")

    query = st.text_input("Enter your question:")
    
    if st.button("Submit") and query:  # Only process if submit button is clicked and query is not empty
        # Format the chat history for the chain
        formatted_chat_history = [(chat["question"], chat.get("answer", "")) for chat in st.session_state.chat_history]
        
        result = qa({"question": query, "chat_history": formatted_chat_history})

        # Update chat history with the new question and answer
        st.session_state.chat_history.append({"question": query, "answer": result["answer"]})

        # Clear the input box after submitting the question
        st.experimental_rerun()
