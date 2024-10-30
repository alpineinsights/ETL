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
from llama_index.embeddings.voyageai import VoyageEmbedding

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
    
    @retry(stop=stop_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
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

def chunk_markdown(text: str, chunk_size: int, overlap: int) -> List[str]:
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

def get_embedding_model(config: Dict) -> VoyageEmbedding:
    """Initialize Voyage embedding model through LlamaIndex"""
    return VoyageEmbedding(
        model_name=config['embedding_model'],
        voyage_api_key=st.secrets['VOYAGE_API_KEY']
    )

def process_single_document(url: str, clients: Dict, index: QdrantIndex, config: Dict):
    """Process a single document using LlamaIndex components"""
    try:
        # Download document
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
            
        try:
            # Parse document
            parsed_docs = clients['llama_parse'].load_data(tmp_path)
            
            # Initialize embedding model
            embed_model = get_embedding_model(config)
            
            for doc in parsed_docs:
                chunks = chunk_markdown(doc.text, config['chunk_size'], config['chunk_overlap'])
                documents = []
                dense_vectors = []
                
                with st.progress(0) as chunk_progress:
                    for i, chunk in enumerate(chunks):
                        # Get context
                        context = get_document_context(
                            clients['anthropic'],
                            chunk,
                            config['context_prompt'],
                            config['context_model']
                        )
                        
                        # Create document
                        documents.append(Document(
                            text=chunk,
                            context=context,
                            url=url,
                            date_processed=datetime.now().isoformat()
                        ))
                        
                        # Get embedding using LlamaIndex
                        vector = embed_model.get_text_embedding(chunk)
                        dense_vectors.append(vector)
                        
                        chunk_progress.progress((i + 1) / len(chunks))
                
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
            metrics = {
                'total_documents': len(urls),
                'processed_documents': 0,
                'total_chunks': 0,
                'failed_documents': 0,
                'start_time': datetime.now()
            }
            
            # Create placeholders for dynamic updates
            overall_progress = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()
            
            for i, url in enumerate(urls):
                try:
                    status_text.text(f"Processing document {i+1}/{len(urls)}: {url}")
                    process_single_document(url, clients, index, config)
                    metrics['processed_documents'] += 1
                except Exception as e:
                    metrics['failed_documents'] += 1
                    st.error(f"Error processing document {url}: {str(e)}")
                
                # Update progress
                overall_progress.progress((i + 1) / len(urls))
                
                # Calculate and display metrics
                elapsed_time = (datetime.now() - metrics['start_time']).total_seconds()
                docs_per_second = metrics['processed_documents'] / elapsed_time if elapsed_time > 0 else 0
                
                with metrics_container:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Processed Documents", metrics['processed_documents'])
                    with col2:
                        st.metric("Failed Documents", metrics['failed_documents'])
                    with col3:
                        st.metric("Processing Rate", f"{docs_per_second:.2f} docs/sec")
            
            # Display final summary
            processing_time = (datetime.now() - metrics['start_time']).total_seconds()
            st.success(f"""
                Processing complete! 
                - Index name: {st.session_state.index_name}
                - Total documents processed: {metrics['processed_documents']}
                - Failed documents: {metrics['failed_documents']}
                - Total processing time: {processing_time:.2f} seconds
                - Average processing rate: {metrics['processed_documents']/processing_time:.2f} docs/sec
            """)
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.exception(e)

def main():
    render_ui()

if __name__ == "__main__":
    main()

Version 2 of 2




