import streamlit as st

# ChromaDB monkey patching, remove when fixed and uncomment to run locally
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from process_document import upload_file, cleanup
from embed_and_retrieve import create_query_engine, validate_api_key, get_logger

import openai
import requests

logger = get_logger()

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

# Initialize session state variables
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Streamlit app
st.title(":speech_balloon: Chat with Your Documents")

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

provider = st.sidebar.selectbox("Select Provider", ["OpenAI", "HuggingFace"])

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
    
    if st.session_state.query_engine:
        st.sidebar.success("Document embedded successfully!")
    else:
        st.sidebar.error("""Failed to embed document. There were issues creating the query engine.
                         (Most likely due to rate limits on HuggingFace/OpenAI LLMs).
                         Please obtain a (new) HF access token at https://hf.co/settings/tokens and try again.
                         If the issue persists, please try again later.""")

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
            # Attempt to stream response, otherwise fall back to standard block output
            try:
                for chunk in response.response_gen:
                    full_response += chunk
                    message_placeholder.markdown(full_response)
            except Exception as e:
                logger.warning(f"Streaming failed: {e}. Falling back to standard block output.")
                response = st.session_state.query_engine.query(prompt)
                full_response = response.response
                message_placeholder.markdown(full_response)
        else:
            full_response = """Please upload and embed a document first. 
            If you have already done so, there was an issue with the query engine.
            (Most likely due to rate limits on HuggingFace/OpenAI LLMs or an invalid API key).
            Please obtain a (new) HF access token at https://hf.co/settings/tokens and try again."""
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Register cleanup function
import atexit
atexit.register(cleanup)