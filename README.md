# chat-with-your-docs
Chat with your Text Documents (upto 25MB in size)

**Tech Stack Used:** 
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader/) for parsing documents
- [ChromaDB](https://docs.trychroma.com/) for vector database storage, ranking and retrieval
- Choice of OpenAI/HuggingFace as the LLM (Large Language Model) for both embedding and inference. Obtain an API key from [here for OpenAI](https://platform.openai.com/api-keys) or [here for HuggingFace](https://hf.co/settings/tokens)
  - Embedding models are `sentence-transformers/all-MiniLM-L6-v2` for HF and `text-embedding-3-small` for OpenAI
  - Inference models are `meta-llama/Meta-Llama-3.1-8B-Instruct` for HF and `gpt-4o-mini` for OpenAI

Try out the App [here](https://chatwithyourdocuments.streamlit.app/)
