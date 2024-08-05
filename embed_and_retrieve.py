import streamlit as st

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import BitsAndBytesConfig
import torch

import chromadb
import openai
from huggingface_hub.utils import HfHubHTTPError
from openai import OpenAIError
import requests

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("chat_with_your_documents")
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s : %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    handlers=[logging.StreamHandler(), RotatingFileHandler(filename="chat_with_your_documents_app.log", maxBytes=5*1024*1024, backupCount=1)])

def get_logger():
    return logger

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

def create_query_engine(file_path, provider, api_key, download_llm=False):
    # Set up the embedding and inference models
    if provider == "OpenAI":
        embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)
        llm = OpenAI(api_key=api_key)
    else:
        if api_key:
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", token=api_key)
            if download_llm:
                llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-beta")
            else:
                llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-beta", token=api_key)
        else:
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            if download_llm:
                llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-beta")
            else:
                llm = HuggingFaceInferenceAPI(model_name="HuggingFaceH4/zephyr-7b-beta")

    
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # Set up the vector store (ChromaDB)
    chroma_client = chromadb.Client()
    collection_name = "document_collection"
    try:
        chroma_collection = chroma_client.create_collection(collection_name)      
    except (ValueError, chromadb.db.base.UniqueConstraintError, chromadb.errors.ChromaError) as e:
        logger.warning(f"{e}: Collection already exists. Deleting and creating a new collection.")
        chroma_client.delete_collection(collection_name)
        chroma_collection = chroma_client.create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create vector store index
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

    # Create and return query engine (attempt streaming support)
    try:
        query_engine = index.as_query_engine(streaming=True, service_context=service_context)
        response = query_engine.query("What is the document about?")
        for text in response.response_gen:
            if text is not None:
                break
    except NotImplementedError as e:
        logger.warning(f"{e}: Streaming not supported. Creating query engine without streaming.")
        try:
            query_engine = index.as_query_engine(service_context=service_context)
            response = query_engine.query("What is the document about?")
        except HfHubHTTPError as e:
            logger.error(f"{e}: Most likely the rate limits of the HuggingFace API have been exceeded. Downloading the LLM")
            try:
                llm = HuggingFaceLLM(model_name="HuggingFaceH4/zephyr-7b-beta")
                service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
                query_engine = index.as_query_engine(service_context=service_context)
                response = query_engine.query("What is the document about?")
            except Exception as e:
                logger.error(f"{e}: Unable to create query engine from HuggingFace.")
                st.error("Unable to create query engine.")
                return None
        except OpenAIError as e:
            logger.error(f"{e}: Most likely the rate limits of the OpenAI API have been exceeded.")
            st.error("Unable to create query engine from OpenAI (Mostly due to rate limiting). Please try after sometime.")
            return None

    return query_engine