import streamlit as st
import requests
import xml.etree.ElementTree as ET
import os
import tempfile
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from tenacity import retry, stop_after_attempt, wait_exponential

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
        
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
        from qdrant_client.http.exceptions import UnexpectedResponse
    except ImportError:
        missing_packages.append("qdrant-client")
    
    if missing_packages:
        st.error(f"""
            Missing required packages: {', '.join(missing_packages)}
            Please check your requirements.txt file.
        """)
        st.stop()
    
    return anthropic, voyageai, LlamaParse, QdrantClient, models, UnexpectedResponse

# Check dependencies before proceeding
anthropic, voyageai, LlamaParse, QdrantClient, qdrant_models, UnexpectedResponse = check_dependencies()

@dataclass
class Document:
    """Document class to store text and metadata"""
    text: str
    context: str
    url: str
    date_processed: str
    company_name: str

class QdrantIndex:
    """Vector store using Qdrant Cloud for hybrid search"""
    def __init__(self, url: str, api_key: str, index_name: str):
        self.index_name = index_name
        
        # Initialize Qdrant Cloud client with retry mechanism
        self.client = self._initialize_client(url, api_key)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),
            max_features=10000
        )
        self.documents: List[Document] = []
        
        # Configure vectors
        self.dense_config = qdrant_models.VectorParams(
            size=768,  # Voyage AI dimension
            distance=qdrant_models.Distance.COSINE
        )
        
        self.sparse_config = qdrant_models.VectorParams(
            size=10000,  # Match TF-IDF max_features
            distance=qdrant_models.Distance.COSINE
        )
        
        self._ensure_collection_exists()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _initialize_client(self, url: str, api_key: str) -> QdrantClient:
        """Initialize Qdrant client with retry mechanism"""
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=100
        )
        # Test connection
        client.get_collections()
        return client
    
    def _ensure_collection_exists(self):
        """Ensure collection exists with proper configuration"""
        try:
            collections = self.client.get_collections()
            if self.index_name not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.index_name,
                    vectors_config={
                        "dense": self.dense_config,
                        "sparse": self.sparse_config
                    },
                    optimizers_config=qdrant_models.OptimizersConfigDiff(
                        indexing_threshold=0,
                        memmap_threshold=10000
                    )
                )
                st.info(f"Created new Qdrant collection: {self.index_name}")
            else:
                st.info(f"Using existing Qdrant collection: {self.index_name}")
        except UnexpectedResponse as e:
            st.error(f"Qdrant Cloud error: {str(e)}")
            raise
        except Exception as e:
            st.error(f"Error initializing Qdrant collection: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def add_documents(self, docs: List[Document], dense_vectors: List[List[float]]):
        """Add documents to Qdrant Cloud with retry mechanism"""
        try:
            # Update documents list
            self.documents.extend(docs)
            
            # Compute sparse vectors
            texts = [doc.text for doc in docs]
            sparse_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Prepare points for Qdrant
            points = []
            for i, (doc, dense_vec) in enumerate(zip(docs, dense_vectors)):
                sparse_vec = sparse_matrix[i].toarray()[0].tolist()
                
                point = qdrant_models.PointStruct(
                    id=str(len(self.documents)-len(docs)+i),
                    vectors={
                        "dense": dense_vec,
                        "sparse": sparse_vec
                    },
                    payload={
                        "text": doc.text,
                        "context": doc.context,
                        "url": doc.url,
                        "date_processed": doc.date_processed,
                        "company_name": doc.company_name
                    }
                )
                points.append(point)
            
            # Upload to Qdrant in batches with progress tracking
            batch_size = 50
            with st.progress(0) as progress_bar:
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.client.upsert(
                        collection_name=self.index_name,
                        points=batch,
                        wait=True
                    )
                    progress = (i + len(batch)) / len(points)
                    progress_bar.progress(progress)
                    time.sleep(0.2)
                
        except Exception as e:
            st.error(f"Error adding documents to Qdrant Cloud: {str(e)}")
            raise

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Extract PDF URLs from XML sitemap recursively"""
    pdf_urls = []
    
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        sitemaps = root.findall('.//ns:sitemap/ns:loc', namespaces)
        if sitemaps:
            for sitemap in sitemaps:
                pdf_urls.extend(get_urls_from_sitemap(sitemap.text))
        else:
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
        if current_length + line_length > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            overlap_text = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
            current_chunk = overlap_text
            current_length = sum(len(line) for line in current_chunk)
        
        current_chunk.append(line)
        current_length += line_length
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_document_context(client: anthropic.Anthropic, content: str, company_name: str) -> str:
    """Generate context for a document chunk using Claude with retry mechanism"""
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

def process_single_document(url: str, clients: Dict, index: QdrantIndex, company_name: str):
    """Process a single document URL with improved error handling"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        try:
            parsed_docs = clients['llama_parse'].load_data(tmp_path)
            
            for doc in parsed_docs:
                chunks = chunk_markdown(doc.text)
                documents = []
                dense_vectors = []
                
                with st.progress(0) as progress_bar:
                    for i, chunk in enumerate(chunks):
                        context = get_document_context(
                            clients['anthropic'],
                            chunk,
                            company_name
                        )
                        
                        doc = Document(
                            text=chunk,
                            context=context,
                            url=url,
                            date_processed=datetime.now().isoformat(),
                            company_name=company_name
                        )
                        documents.append(doc)
                        
                        vector = clients['voyage'].embed(
                            [chunk],
                            model=st.session_state.get('voyage_model', 'voyage-finance-2')
                        )[0]
                        dense_vectors.append(vector)
                        
                        progress = (i + 1) / len(chunks)
                        progress_bar.progress(progress)
                        time.sleep(0.1)
                
                index.add_documents(documents, dense_vectors)
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")

def render_ui():
    """Render the Streamlit UI"""
    st.title("📚 Document Processing Pipeline")
    
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'anthropic': '',
            'voyage': '',
            'llamaparse': '',
            'qdrant': ''
        }
    
    with st.expander("🔑 API Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.api_keys['anthropic'] = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.api_keys['anthropic']
            )
            st.session_state.api_keys['voyage'] = st.text_input(
                "Voyage AI API Key",
                type="password",
                value=st.session_state.api_keys['voyage']
            )
        with col2:
            st.session_state.api_keys['llamaparse'] = st.text_input(
                "LlamaParse API Key",
                type="password",
                value=st.session_state.api_keys['llamaparse']
            )
            st.session_state.api_keys['qdrant'] = st.text_input(
                "Qdrant Cloud API Key",
                type="password",
                value=st.session_state.api_keys['qdrant']
            )
        
        st.session_state['qdrant_url'] = st.text_input(
            "Qdrant Cloud URL",
            value=st.session_state.get('qdrant_url', ''),
            help="Format: https://your-cluster-name.qdrant.tech"
        )
    
    with st.expander("⚙️ Processing Configuration", expanded=True):
        st.session_state.sitemap_url = st.text_input("XML Sitemap URL")
        st.session_state.company_name = st.text_input("Company Name")
        st.session_state.voyage_model = st.selectbox(
            "Voyage AI Model",
            options=["voyage-finance-2", "voyage-2"],
            index=0
        )
        st.session_state.index_name = f"hybrid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if st.button("🚀 Start Processing"):
        if not all(st.session_state.api_keys.values()):
            st.error("Please provide all API keys")
            return
        
        if not st.session_state['qdrant_url']:
            st.error("Please provide the Qdrant Cloud URL")
            return
        
        if not st.session_state.sitemap_url:
            st.error("Please provide a sitemap URL")
            return
        
        if not st.session_state.company_name:
            st.error("Please provide a company name")
            return
        
        try:
            clients = {
                'anthropic': anthropic.Anthropic(api_key=st.session_state.api_keys['anthropic']),
                'voyage': voyageai.Client(api_key=st.session_state.api_keys['voyage']),
                'llama_parse': LlamaParse(api_key=st.session_state.api_keys['llamaparse'])
            }
            
            index = QdrantIndex(
                url=st.session_state['qdrant_url'],
                api_key=st.session_state.api_keys['qdrant'],
                index_name=st.session_state.index_name
            )
            
            urls = get_urls_from_sitemap(st.session_state.sitemap_url)
            st.info(f"Found {len(urls)} PDF documents")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, url in enumerate(urls):
                status_text.text(f"Processing {url}")
                process_single_document(url, clients, index, st.session_state.company_name)
                progress_bar.progress((i + 1) / len(urls))
            
            st.success(f"Processing complete! Index name: {st.session_state.index_name}")
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.exception(e)

def main():
    render_ui()

if __name__ == "__main__":
    main()
