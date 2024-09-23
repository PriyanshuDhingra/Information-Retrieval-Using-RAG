from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from QAWithPdf.data_ingestion import load_data
from QAWithPdf.model_api import load_model
import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, documents):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Parameters:
    - model: The loaded Gemini model.
    - documents (list): A list of documents to vectorize.

    Returns:
    - query_engine: A query engine to search over vectorized documents.
    """
    try:
        logging.info("Initializing Gemini embedding model...")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")
        # service_context = ServiceContext.from_defaults(
        #     llm=model,
        #     embed_model=gemini_embed_model,
        #     chunk_size=800,
        #     chunk_overlap=20
        # )
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        transformations = [SentenceSplitter(chunk_size=1000, chunk_overlap=20)]
        logging.info("Vectorizing documents into embeddings...")
        index = VectorStoreIndex.from_documents(documents, llm = model, embed_model = gemini_embed_model, transformations=transformations)
        index.storage_context.persist()
        
        logging.info("Embedding completed. Creating query engine...")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        raise customexception(e, sys)
