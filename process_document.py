import os
import tempfile
import shutil
import logging

import chromadb
import chromadb.errors as chromadb_errors

from embed_and_retrieve import get_logger

logger = get_logger()

# Create a temporary directory to store uploaded files
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def upload_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}", dir=UPLOAD_DIR) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def cleanup():
    # Remove temporary files
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    
    # Remove ChromaDB collection
    import chromadb
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("document_collection")
    except (ValueError, chromadb_errors.ChromaError) as e:
        logger.warning(f"Failed to delete collection: {e}. Ignoring exception.")