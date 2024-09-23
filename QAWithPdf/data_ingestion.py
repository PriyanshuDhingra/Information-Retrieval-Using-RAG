from llama_index.core import SimpleDirectoryReader
import os
import sys
from exception import customexception
from logger import logging

def load_data(file_paths):
    """
    Load PDF documents from the paths of uploaded files.

    Parameters:
    - file_paths (list): List of file paths of uploaded PDFs.

    Returns:
    - List of loaded PDF documents.
    """
    try:
        logging.info("Data loading started...")
        
        # Initialize a SimpleDirectoryReader with the temporary directory
        directory = os.path.dirname(file_paths[0])  # Assuming all files are in the same directory
        loader = SimpleDirectoryReader(directory)
        documents = loader.load_data()

        logging.info("Data loading completed.")
        return documents

    except Exception as e:
        logging.error("Exception in loading data.")
        raise customexception(e, sys)
