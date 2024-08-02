import streamlit as st

# ChromaDB monkey patching
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from process_document import upload_file, cleanup
from embed_and_retrieve import create_query_engine, validate_api_key

import chromadb
import chromadb.errors as chromadb_errors
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

import openai
import requests
import logging

logger = logging.getLogger()

def validate_api_key(provider, api_key):
    if provider == "OpenAI":
        try:
            url = "https://api.openai.com/v1/engines"
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            response = requests.get(url, headers=headers)
            return response.status_code == 200
        except openai.OpenAIError as e:
            return False
    elif provider == "HuggingFace":
        if not api_key:
            return True
        try:
            response = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            return response.status_code == 200
        except:
            return False

def create_query_engine(file_path, provider, api_key):
    # Set up the embedding model
    if provider == "OpenAI":
        embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
        llm = OpenAI(api_key=api_key)
    else:
        if api_key:
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", token=api_key)
            llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-beta", token=api_key)
        else:
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-beta")

    # Load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # Set up ChromaDB
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("document_collection")
    except chromadb_errors.ChromaError as e:
        logger.warning(f"Failed to delete collection: {e}")
    chroma_collection = chroma_client.get_or_create_collection("document_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

    # Create query engine
    query_engine = index.as_query_engine(service_context=service_context)

    # Return query engine and document count
    document_count = chroma_collection.count()
    return query_engine, document_count

# Initialize session state variables
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit app
st.title("Chat with Your Documents")

# Sidebar
st.sidebar.header("Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a document (Max size 25MB)", type=["pdf", "docx", "pptx", "csv"])

# Validate file upload and ensure it is a supported file type less than or equal to 25MB
if uploaded_file:
    if uploaded_file.size > 25 * 1024 * 1024:
        st.sidebar.error("File size exceeds 25MB limit!")
        uploaded_file = None
    elif uploaded_file.type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "text/csv"]:
        st.sidebar.error("Unsupported file type!")
        uploaded_file = None

# Provider selection
provider = st.sidebar.selectbox("Select Provider", ["OpenAI", "HuggingFace"])

# API key input
api_key = st.sidebar.text_input("Enter API Key", type="password")

# Validate API key when entered
if api_key:
    if validate_api_key(provider, api_key):
        st.sidebar.success("API key is valid!")
    else:
        st.sidebar.error("Invalid API key!")

# Enable/disable embed button
embed_button = st.sidebar.button("Embed", disabled=not (uploaded_file and (validate_api_key(provider, api_key) or (provider == "HuggingFace" and not api_key))))

# Embed document when button is clicked
if embed_button and uploaded_file:
    with st.spinner("Embedding document..."):
        file_path = upload_file(uploaded_file)
        st.session_state.query_engine = create_query_engine(file_path, provider, api_key)
        st.session_state.messages = []  # Clear chat history
    
    st.sidebar.success("Document embedded successfully!")

# Chat interface
st.subheader("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document", disabled=not st.session_state.query_engine):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if st.session_state.query_engine:
            response = st.session_state.query_engine.query(prompt)
            full_response = response.response
            # if response.get_formatted_sources():
            #     full_response += f"\n\nSources used: {response.get_formatted_sources()}"
            message_placeholder.markdown(full_response)
        else:
            full_response = "Please upload and embed a document first."
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Register cleanup function
import atexit
atexit.register(cleanup)