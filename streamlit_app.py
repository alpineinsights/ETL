import streamlit as st
import requests
import xml.etree.ElementTree as ET
import os
import tempfile
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Import required packages
try:
    import anthropic
    import voyageai
    from llama_parse import LlamaParse
except ImportError:
    st.error("""
        Please install required packages by creating a requirements.txt file with the following content:
        
        anthropic==0.8.1
        voyageai==0.1.4
        llama-parse==0.1.1
        scikit-learn==1.3.2
        numpy==1.26.4
        tqdm==4.66.2
        requests==2.31.0
        streamlit==1.32.2
        
        Then deploy this app to Streamlit Cloud.
    """)
    st.stop()

# Set page config
st.set_page_config(
    page_title="Document Processing Pipeline",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        .stAlert > div {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .stMarkdown {
            padding: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

@dataclass
class Document:
    """Document class to store text and metadata"""
    text: str
    context: str
    url: str
    date_processed: str

class QuadrantIndex:
    """Vector store combining TF-IDF and dense vectors"""
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.dense_vectors = []
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2)
        )
        self.documents: List[Document] = []
        self.sparse_matrix = None
        
    def add_documents(self, docs: List[Document], dense_vectors: List[List[float]]):
        """Add a batch of documents to both dense and sparse indices"""
        # Update dense vectors
        self.dense_vectors.extend(dense_vectors)
        
        # Update documents list
        self.documents.extend(docs)
        
        # Recompute TF-IDF matrix for all documents
        texts = [doc.text for doc in self.documents]
        self.sparse_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def save_to_session_state(self):
        """Save index to Streamlit session state"""
        st.session_state['index_data'] = {
            'name': self.index_name,
            'dense_vectors': self.dense_vectors,
            'documents': self.documents,
            'tfidf_vectorizer': pickle.dumps(self.tfidf_vectorizer),
            'sparse_matrix': {
                'data': self.sparse_matrix.data,
                'indices': self.sparse_matrix.indices,
                'indptr': self.sparse_matrix.indptr,
                'shape': self.sparse_matrix.shape
            } if self.sparse_matrix is not None else None
        }

    @classmethod
    def load_from_session_state(cls):
        """Load index from Streamlit session state"""
        if 'index_data' not in st.session_state:
            return None
        
        data = st.session_state['index_data']
        index = cls(data['name'])
        index.dense_vectors = data['dense_vectors']
        index.documents = data['documents']
        index.tfidf_vectorizer = pickle.loads(data['tfidf_vectorizer'])
        
        if data['sparse_matrix']:
            from scipy.sparse import csr_matrix
            index.sparse_matrix = csr_matrix(
                (data['sparse_matrix']['data'],
                 data['sparse_matrix']['indices'],
                 data['sparse_matrix']['indptr']),
                shape=data['sparse_matrix']['shape']
            )
        
        return index

def process_documents():
    """Main document processing function"""
    # Initialize clients
    try:
        clients = {
            'anthropic': anthropic.Anthropic(api_key=st.session_state.api_keys['anthropic']),
            'voyage': voyageai.Client(api_key=st.session_state.api_keys['voyage']),
            'llama_parse': LlamaParse(api_key=st.session_state.api_keys['llamaparse'])
        }
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return
    
    # Initialize index
    index = QuadrantIndex(st.session_state.index_name)
    
    # Process URLs
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Get PDF URLs
        urls = get_urls_from_sitemap(st.session_state.sitemap_url)
        st.info(f"Found {len(urls)} PDF documents")
        
        for i, url in enumerate(urls):
            status_text.text(f"Processing {url}")
            process_single_document(url, clients, index)
            progress_bar.progress((i + 1) / len(urls))
        
        # Save index to session state
        index.save_to_session_state()
        st.success("Processing complete!")
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.exception(e)

def render_ui():
    """Render the Streamlit UI"""
    st.title("📚 Document Processing Pipeline")
    
    # Initialize session state
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'anthropic': '',
            'voyage': '',
            'llamaparse': ''
        }
    
    # API Configuration
    with st.expander("🔑 API Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.api_keys['anthropic'] = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.api_keys['anthropic']
            )
        with col2:
            st.session_state.api_keys['voyage'] = st.text_input(
                "Voyage AI API Key",
                type="password",
                value=st.session_state.api_keys['voyage']
            )
        with col3:
            st.session_state.api_keys['llamaparse'] = st.text_input(
                "LlamaParse API Key",
                type="password",
                value=st.session_state.api_keys['llamaparse']
            )
    
    # Processing Configuration
    with st.expander("⚙️ Processing Configuration", expanded=True):
        st.session_state.sitemap_url = st.text_input("XML Sitemap URL")
        st.session_state.company_name = st.text_input("Company Name")
        st.session_state.index_name = st.text_input("Index Name", "hybrid_search")
        st.session_state.voyage_model = st.selectbox(
            "Voyage AI Model",
            options=["voyage-finance-2", "voyage-2"],
            index=0
        )
        st.session_state.batch_size = st.number_input(
            "Processing Batch Size",
            min_value=1,
            value=5
        )
    
    # Process button
    if st.button("🚀 Start Processing"):
        if not all(st.session_state.api_keys.values()):
            st.error("Please provide all API keys")
            return
        
        if not st.session_state.sitemap_url:
            st.error("Please provide a sitemap URL")
            return
        
        process_documents()

def main():
    render_ui()

if __name__ == "__main__":
    main()
