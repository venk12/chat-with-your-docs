from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext

from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb
import chromadb.errors as chromadb_errors
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
    collection_name = "document_collection"
    try:
        chroma_collection = chroma_client.create_collection(collection_name)      
    except (ValueError, chromadb_errors.ChromaError) as e:
        logger.info(f"{e}: Collection already exists. Deleting and creating a new collection.")
        chroma_client.delete_collection(collection_name)
        chroma_collection = chroma_client.create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, service_context=service_context)

    # Create and return query engine and document count
    query_engine = index.as_query_engine(service_context=service_context)
    return query_engine