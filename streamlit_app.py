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
import time

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Document Processing Pipeline",
    page_icon="📚",
    layout="wide"
)

# Add dependencies check
def check_dependencies():
    missing_packages = []
    try:
        import anthropic
    except ImportError:
        missing_packages.append("anthropic")
    
    try:
        import voyageai
    except ImportError:
        missing_packages.append("voyageai")
        
    try:
        from llama_parse import LlamaParse
    except ImportError:
        missing_packages.append("llama-parse")
    
    if missing_packages:
        st.error(f"""
            Missing required packages: {', '.join(missing_packages)}
            Please check your requirements.txt file.
        """)
        st.stop()
    
    return anthropic, voyageai, LlamaParse

# Check dependencies before proceeding
anthropic, voyageai, LlamaParse = check_dependencies()

@dataclass
class Document:
    """Document class to store text and metadata"""
    text: str
    context: str
    url: str
    date_processed: str
    company_name: str

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

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Extract PDF URLs from XML sitemap recursively"""
    pdf_urls = []
    
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        # Handle both sitemap index and regular sitemaps
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Check if this is a sitemap index
        sitemaps = root.findall('.//ns:sitemap/ns:loc', namespaces)
        if sitemaps:
            for sitemap in sitemaps:
                pdf_urls.extend(get_urls_from_sitemap(sitemap.text))
        else:
            # Regular sitemap - extract PDF URLs
            urls = root.findall('.//ns:url/ns:loc', namespaces)
            pdf_urls.extend([url.text for url in urls if url.text.lower().endswith('.pdf')])
    
    except Exception as e:
        st.error(f"Error processing sitemap {sitemap_url}: {str(e)}")
    
    return pdf_urls

def chunk_markdown(text: str, chunk_size: int = 1024, overlap: int = 200) -> List[str]:
    """Chunk markdown text while preserving structure"""
    chunks = []
    lines = text.split('\n')
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line_length = len(line)
        
        # Start new chunk if adding this line would exceed chunk size
        if current_length + line_length > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            # Keep last few lines for overlap
            overlap_text = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
            current_chunk = overlap_text
            current_length = sum(len(line) for line in current_chunk)
        
        current_chunk.append(line)
        current_length += line_length
    
    # Add final chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def get_document_context(client: anthropic.Anthropic, content: str, company_name: str) -> str:
    """Generate context for a document chunk using Claude"""
    prompt = f"""
    Company Name: {company_name}
    
    Document Content:
    {content}
    
    Please provide a brief context for this document chunk, including:
    1. The company name
    2. The apparent date of the document
    3. Any fiscal period mentioned
    4. The main topic or purpose of this content
    
    Provide only the contextual summary, nothing else.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Error generating context: {str(e)}")
        return ""

def process_single_document(url: str, clients: Dict, index: QuadrantIndex, company_name: str):
    """Process a single document URL"""
    try:
        # Download PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        try:
            # Parse PDF
            parsed_docs = clients['llama_parse'].load_data(tmp_path)
            
            for doc in parsed_docs:
                # Chunk the document
                chunks = chunk_markdown(doc.text)
                
                # Process chunks
                documents = []
                dense_vectors = []
                
                for chunk in chunks:
                    # Get context
                    context = get_document_context(
                        clients['anthropic'],
                        chunk,
                        company_name
                    )
                    
                    # Create document
                    doc = Document(
                        text=chunk,
                        context=context,
                        url=url,
                        date_processed=datetime.now().isoformat(),
                        company_name=company_name
                    )
                    documents.append(doc)
                    
                    # Get dense embedding
                    vector = clients['voyage'].embed(
                        [chunk],
                        model=st.session_state.get('voyage_model', 'voyage-finance-2')
                    )[0]
                    dense_vectors.append(vector)
                    
                    # Add small delay to avoid rate limits
                    time.sleep(0.1)
                
                # Add batch to index
                index.add_documents(documents, dense_vectors)
        
        finally:
            # Cleanup temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")

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
    
    # Process button
    if st.button("🚀 Start Processing"):
        if not all(st.session_state.api_keys.values()):
            st.error("Please provide all API keys")
            return
        
        if not st.session_state.sitemap_url:
            st.error("Please provide a sitemap URL")
            return
        
        if not st.session_state.company_name:
            st.error("Please provide a company name")
            return
        
        try:
            # Initialize clients
            clients = {
                'anthropic': anthropic.Anthropic(api_key=st.session_state.api_keys['anthropic']),
                'voyage': voyageai.Client(api_key=st.session_state.api_keys['voyage']),
                'llama_parse': LlamaParse(api_key=st.session_state.api_keys['llamaparse'])
            }
            
            # Initialize index
            index = QuadrantIndex(st.session_state.index_name)
            
            # Get PDF URLs
            urls = get_urls_from_sitemap(st.session_state.sitemap_url)
            st.info(f"Found {len(urls)} PDF documents")
            
            # Process documents with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, url in enumerate(urls):
                status_text.text(f"Processing {url}")
                process_single_document(url, clients, index, st.session_state.company_name)
                progress_bar.progress((i + 1) / len(urls))
            
            # Save index to session state
            index.save_to_session_state()
            st.success("Processing complete!")
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.exception(e)

def main():
    render_ui()

if __name__ == "__main__":
    main()
