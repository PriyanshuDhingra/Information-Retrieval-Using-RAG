import streamlit as st
from QAWithPdf.data_ingestion import load_data
from QAWithPdf.embedding import download_gemini_embedding
from QAWithPdf.model_api import load_model
import os

def save_uploaded_file(uploaded_file):
    try:
        # Save uploaded file to a temporary location
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("temp", uploaded_file.name)
    except Exception as e:
        return None

def main():
    st.set_page_config("QA with Documents")
    
    uploaded_files = st.file_uploader("Upload your document(s)", accept_multiple_files=True)
    
    st.header("QA with Documents (Information Retrieval)")
    
    user_question = st.text_input("Ask your question")
    
    if st.button("Submit & Process"):
        if uploaded_files is not None:
            with st.spinner("Processing..."):
                # Save each uploaded file
                saved_files = []
                for uploaded_file in uploaded_files:
                    saved_file_path = save_uploaded_file(uploaded_file)
                    if saved_file_path:
                        saved_files.append(saved_file_path)
                
                # Process documents only from uploaded files
                documents = load_data(saved_files)
                model = load_model()
                query_engine = download_gemini_embedding(model, documents)
                
                response = query_engine.query(user_question)
                
                st.write(response.response)

if __name__ == "__main__":
    main()
