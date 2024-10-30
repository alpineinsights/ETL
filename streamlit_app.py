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
from tenacity import retry, stop_after_attempt, wait_exponential

# New imports for LlamaIndex
from llama_index.core import SimpleDirectoryReader, Document as LlamaDocument
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings import VoyageEmbedding
from llama_index.core import download_loader

# Default context prompt
DEFAULT_CONTEXT_PROMPT = """
Please analyze this document chunk and provide a brief contextual summary that includes:
1. The apparent date of the document
2. Any fiscal period mentioned
3. The main topic or purpose of this content

Provide only the contextual summary, nothing else.
"""

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Document Processing Pipeline",
    page_icon="📚",
    layout="wide"
)

# Check for required API keys in Streamlit secrets
REQUIRED_KEYS = ['ANTHROPIC_API_KEY', 'VOYAGE_API_KEY', 'LLAMA_PARSE_API_KEY', 'QDRANT_API_KEY']
if not all(key in st.secrets for key in REQUIRED_KEYS):
    st.error(f"""
        Missing required API keys in Streamlit secrets.
        Please add the following in your Streamlit Cloud dashboard under Settings -> Secrets:
        {', '.join(REQUIRED_KEYS)}
    """)
    st.stop()

# Add dependencies check
def check_dependencies():
    required_packages = {
        'anthropic': 'anthropic',
        'voyageai': 'voyageai',
        'llama_parse': 'llama-parse',
        'llama_index': 'llama-index-core',
        'qdrant_client': 'qdrant-client'
    }
    
    missing_packages = []
    modules = {}
    
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'anthropic':
                import anthropic
                modules['anthropic'] = anthropic
            elif module_name == 'voyageai':
                from voyageai import Client
                modules['VoyageClient'] = Client
            elif module_name == 'llama_parse':
                from llama_parse import LlamaParse
                modules['LlamaParse'] = LlamaParse
            elif module_name == 'qdrant_client':
                from qdrant_client import QdrantClient
                from qdrant_client.http import models
                from qdrant_client.http.exceptions import UnexpectedResponse
                modules.update({
                    'QdrantClient': QdrantClient,
                    'qdrant_models': models,
                    'UnexpectedResponse': UnexpectedResponse
                })
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        st.error(f"""
            Missing required packages: {', '.join(missing_packages)}
            Please check your requirements.txt file.
        """)
        st.stop()
    
    return modules

# Check dependencies before proceeding
modules = check_dependencies()
globals().update(modules)

@dataclass
class Document:
    """Document class to store text and metadata"""
    text: str
    context: str
    url: str
    date_processed: str

class QdrantIndex:
    """Vector store using Qdrant Cloud for hybrid search"""
    def __init__(self, url: str, api_key: str, index_name: str):
        self.index_name = index_name
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=100
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),
            max_features=10000
        )
        self.documents: List[Document] = []
        
        # Define vector configurations
        self.dense_config = qdrant_models.VectorParams(
            size=768,  # Voyage AI dimension
            distance=qdrant_models.Distance.COSINE
        )
        
        self.sparse_config = qdrant_models.VectorParams(
            size=10000,  # Match TF-IDF max_features
            distance=qdrant_models.Distance.COSINE
        )
        
        self._ensure_collection_exists()
    
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
        except Exception as e:
            st.error(f"Error initializing Qdrant collection: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def add_documents(self, docs: List[Document], dense_vectors: List[List[float]]):
        """Add documents to Qdrant Cloud with retry mechanism"""
        try:
            self.documents.extend(docs)
            texts = [doc.text for doc in docs]
            sparse_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
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
                        "date_processed": doc.date_processed
                    }
                )
                points.append(point)
            
            batch_size = 50
            with st.progress(0) as batch_progress:
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    self.client.upsert(
                        collection_name=self.index_name,
                        points=batch,
                        wait=True
                    )
                    batch_progress.progress((i + len(batch)) / len(points))
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_document_context(client: anthropic.Anthropic, content: str, prompt: str, model: str) -> str:
    """Generate context for a document chunk using Claude with prompt caching"""
    try:
        response = client.beta.prompt_caching.messages.create(
            model=model,
            max_tokens=200,
            temperature=0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": "\n\nDocument Content:\n" + content
                        }
                    ]
                }
            ],
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Error generating context: {str(e)}")
        return ""

def process_single_document(url: str, clients: Dict, index: QdrantIndex, config: Dict):
    """Process a single document using LlamaIndex components"""
    try:
        # Download PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
            
        try:
            # Load and parse document
            PDFReader = download_loader("PDFReader")
            loader = PDFReader()
            raw_docs = loader.load_data(tmp_path)
            
            # Set up components
            parser = SimpleNodeParser.from_defaults(
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap']
            )
            
            embed_model = VoyageEmbedding(
                api_key=st.secrets['VOYAGE_API_KEY'],
                model_name=config['embedding_model']
            )
            
            documents = []
            dense_vectors = []
            
            with st.progress(0) as chunk_progress:
                # Process each document
                for doc in raw_docs:
                    # Split into chunks
                    nodes = parser.get_nodes_from_documents([doc])
                    chunks = [node.text for node in nodes]
                    
                    for i, chunk in enumerate(chunks):
                        # Get context
                        context = get_document_context(
                            clients['anthropic'],
                            chunk,
                            config['context_prompt'],
                            config['context_model']
                        )
                        
                        # Create document
                        processed_doc = Document(
                            text=chunk,
                            context=context,
                            url=url,
                            date_processed=datetime.now().isoformat()
                        )
                        documents.append(processed_doc)
                        
                        # Get embedding
                        vector = embed_model.get_text_embedding(chunk)
                        dense_vectors.append(vector)
                        
                        chunk_progress.progress((i + 1) / len(chunks))
                        time.sleep(0.1)
                    
                # Add to index
                index.add_documents(documents, dense_vectors)
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")

def render_ui():
    """Render the Streamlit UI"""
    st.title("📚 Document Processing Pipeline")
    
    with st.expander("⚙️ Configuration", expanded=True):
        st.session_state.sitemap_url = st.text_input("XML Sitemap URL")
        
        col1, col2 = st.columns(2)
        with col1:
            config = {
                'embedding_model': st.selectbox(
                    "Embedding Model",
                    options=["voyage-finance-2", "voyage-2"],
                    index=0
                ),
                'chunk_size': st.number_input(
                    "Chunk Size",
                    value=1024,
                    min_value=100,
                    max_value=8192
                ),
                'chunk_overlap': st.number_input(
                    "Chunk Overlap",
                    value=200,
                    min_value=0,
                    max_value=1000
                )
            }
        with col2:
            config.update({
                'context_model': st.selectbox(
                    "Context Model",
                    options=["claude-3-haiku-20240307", "claude-3-sonnet-20240229"],
                    index=0
                ),
                'qdrant_cluster': st.text_input(
                    "Qdrant Cluster URL",
                    value=st.session_state.get('qdrant_url', ''),
                    help="Format: https://your-cluster-name.qdrant.tech",
                    placeholder="https://your-cluster-name.qdrant.tech"
                )
            })
        
        config['context_prompt'] = st.text_area(
            "Context Prompt",
            value=DEFAULT_CONTEXT_PROMPT,
            height=200
        )
        
        st.session_state.index_name = f"hybrid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if st.button("🚀 Start Processing"):
        if not st.session_state.sitemap_url:
            st.error("Please provide a sitemap URL")
            return
            
        if not config['qdrant_cluster']:
            st.error("Please provide the Qdrant Cluster URL")
            return
        
        try:
            # Initialize clients
            clients = {
                'anthropic': anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY']),
                'voyage': VoyageClient(api_key=st.secrets['VOYAGE_API_KEY']),
                'llama_parse': LlamaParse(api_key=st.secrets['LLAMA_PARSE_API_KEY'])
            }
            
            # Initialize Qdrant index
            # Initialize Qdrant index
            index = QdrantIndex(
                url=config['qdrant_cluster'],
                api_key=st.secrets['QDRANT_API_KEY'],
                index_name=st.session_state.index_name
            )
            
            urls = get_urls_from_sitemap(st.session_state.sitemap_url)
            if not urls:
                st.error("No PDF URLs found in the sitemap")
                return
                
            st.info(f"Found {len(urls)} PDF documents")
            
            # Cache status metrics
            cache_metrics = {
                'total_input_tokens': 0,
                'cache_read_tokens': 0,
                'cache_saved_cost': 0
            }
            
            overall_progress = st.progress(0)
            status_text = st.empty()
            metrics_text = st.empty()
            
            for i, url in enumerate(urls):
                status_text.text(f"Processing document {i+1}/{len(urls)}: {url}")
                process_single_document(url, clients, index, config)
                overall_progress.progress((i + 1) / len(urls))
                
                # Update and display cache metrics
                if hasattr(clients['anthropic'], 'last_response_metrics'):
                    metrics = clients['anthropic'].last_response_metrics
                    cache_metrics['total_input_tokens'] += metrics.get('input_tokens', 0)
                    cache_metrics['cache_read_tokens'] += metrics.get('cache_read_tokens', 0)
                    
                    # Calculate approximate cost savings (assuming $0.01 per 1K tokens)
                    saved_tokens = cache_metrics['cache_read_tokens'] * 0.9  # 90% discount on cached tokens
                    cache_metrics['cache_saved_cost'] = (saved_tokens / 1000) * 0.01
                    
                    metrics_text.text(
                        f"Cache Statistics:\n"
                        f"Total Input Tokens: {cache_metrics['total_input_tokens']}\n"
                        f"Cached Tokens Used: {cache_metrics['cache_read_tokens']}\n"
                        f"Estimated Cost Savings: ${cache_metrics['cache_saved_cost']:.2f}"
                    )
            
            st.success(f"""
                Processing complete! 
                Index name: {st.session_state.index_name}
                Total documents processed: {len(urls)}
                Total cost savings from caching: ${cache_metrics['cache_saved_cost']:.2f}
            """)
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.exception(e)

def main():
    render_ui()

if __name__ == "__main__":
    main()
