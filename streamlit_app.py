# streamlit_app.py

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
import time
from sklearn.feature_extraction.text import HashingVectorizer
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, sleep_and_retry
from python_dateutil import parser
import psutil
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
required_api_keys = ['ANTHROPIC_API_KEY', 'VOYAGE_API_KEY', 'LLAMA_PARSE_API_KEY', 'QDRANT_API_KEY']
missing_keys = [key for key not in st.secrets for key in required_api_keys]
if missing_keys:
    st.error(f"""
        Missing required API keys in Streamlit secrets:
        {', '.join(missing_keys)}
        Please add them in your Streamlit Cloud dashboard under Settings -> Secrets.
    """)
    st.stop()

# Add dependencies check
def check_dependencies():
    missing_packages = []
    try:
        from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    except ImportError:
        missing_packages.append("anthropic")
    try:
        from voyage_ai import Client  # Correct import
    except ImportError:
        missing_packages.append("voyage-ai")
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
    return Anthropic, Client, LlamaParse, QdrantClient, models, UnexpectedResponse

# Check dependencies before proceeding
AnthropicClient, VoyageClient, LlamaParse, QdrantClient, qdrant_models, UnexpectedResponse = check_dependencies()

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

        self.hashing_vectorizer = HashingVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),
            n_features=10000
        )
        self.documents: List[Document] = []

        # Define vector configurations
        self.dense_config = qdrant_models.VectorParams(
            size=768,  # Voyage AI dimension
            distance=qdrant_models.Distance.COSINE
        )

        self.sparse_config = qdrant_models.VectorParams(
            size=10000,  # Match HashingVectorizer n_features
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
            logger.exception("Exception occurred while initializing Qdrant collection")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def add_documents(self, docs: List[Document], dense_vectors: List[List[float]]):
        """Add documents to Qdrant Cloud with retry mechanism"""
        try:
            self.documents.extend(docs)
            texts = [doc.text for doc in docs]
            sparse_matrix = self.hashing_vectorizer.transform(texts)

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
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.index_name,
                    points=batch,
                    wait=True
                )
                time.sleep(0.2)

        except Exception as e:
            st.error(f"Error adding documents to Qdrant Cloud: {str(e)}")
            logger.exception("Exception occurred while adding documents to Qdrant")
            raise

def is_valid_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def check_memory_usage():
    """Monitor memory usage and warn if too high"""
    memory_percent = psutil.Process().memory_percent()
    if memory_percent > 80:
        st.warning(f"High memory usage detected: {memory_percent:.1f}%")
        logger.warning(f"High memory usage: {memory_percent:.1f}%")
        return True
    return False

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Extract PDF URLs from XML sitemap recursively"""
    if not is_valid_url(sitemap_url):
        raise ValueError("Invalid sitemap URL format")

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

    except requests.Timeout:
        st.error(f"Timeout while fetching sitemap: {sitemap_url}")
        logger.error(f"Timeout while fetching sitemap: {sitemap_url}")
    except requests.RequestException as e:
        st.error(f"Network error processing sitemap {sitemap_url}: {str(e)}")
        logger.error(f"Network error processing sitemap {sitemap_url}: {str(e)}")
    except Exception as e:
        st.error(f"Error processing sitemap {sitemap_url}: {str(e)}")
        logger.exception(f"Exception occurred while processing sitemap {sitemap_url}")

    return pdf_urls

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text into pieces of specified token length with overlap."""
    words = text.split()
    chunks = []
    i = 0
    total_words = len(words)
    while i < total_words:
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

@sleep_and_retry
@limits(calls=50, period=60)  # 50 calls per minute
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_document_context(client: AnthropicClient, content: str, prompt: str, model: str) -> str:
    """Generate context for a document chunk using Claude with retry mechanism"""
    try:
        response = client.completions.create(
            model=model,
            max_tokens_to_sample=200,
            temperature=0,
            prompt=f"{HUMAN_PROMPT} {prompt}\n\nDocument Content:\n{content}{AI_PROMPT}"
        )
        return response.completion.strip()
    except Exception as e:
        st.error(f"Error generating context: {str(e)}")
        logger.exception("Exception occurred while generating context")
        raise

def process_single_document(url: str, clients: Dict, index: QdrantIndex, config: Dict):
    """Process a single document URL with improved error handling and logging"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        try:
            parsed_docs = clients['llama_parse'].load_data(tmp_path)

            for parsed_doc in parsed_docs:
                chunks = chunk_text(parsed_doc.text, config['chunk_size'], config['chunk_overlap'])
                documents = []
                dense_vectors = []

                for i, chunk in enumerate(chunks):
                    context = get_document_context(
                        clients['anthropic'],
                        chunk,
                        config['context_prompt'],
                        config['context_model']
                    )

                    doc = Document(
                        text=chunk,
                        context=context,
                        url=url,
                        date_processed=datetime.now().isoformat()
                    )
                    documents.append(doc)

                    vector = clients['voyage'].embed(
                        [chunk],
                        model=config['embedding_model']
                    )[0]
                    dense_vectors.append(vector)

                index.add_documents(documents, dense_vectors)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        st.error(f"Error processing {url}: {str(e)}")
        logger.exception(f"Exception occurred while processing {url}")

def render_ui():
    """Render the Streamlit UI"""
    st.title("📚 Document Processing Pipeline")

    with st.expander("⚙️ Configuration", expanded=True):
        st.session_state['sitemap_url'] = st.text_input("XML Sitemap URL")

        col1, col2 = st.columns(2)
        with col1:
            config = {
                'embedding_model': st.selectbox(
                    "Embedding Model",
                    options=["voyage-finance-2", "voyage-2"],
                    index=0
                ),
                'chunk_size': st.number_input(
                    "Chunk Size (number of tokens)",
                    value=512,
                    min_value=100,
                    max_value=8192
                ),
                'chunk_overlap': st.number_input(
                    "Chunk Overlap (number of tokens)",
                    value=50,
                    min_value=0,
                    max_value=1000
                )
            }
        with col2:
            config.update({
                'context_model': st.selectbox(
                    "Context Model",
                    options=["claude-2", "claude-instant-1"],  # Update model names accordingly
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

        config['index_name'] = st.text_input(
            "Index Name",
            value=f"hybrid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Specify a name for the Qdrant index"
        )

    if st.button("🚀 Start Processing"):
        if not st.session_state['sitemap_url']:
            st.error("Please provide a sitemap URL")
            return

        if not config['qdrant_cluster']:
            st.error("Please provide the Qdrant Cluster URL")
            return

        try:
            # Initialize clients using Streamlit secrets
            clients = {
                'anthropic': AnthropicClient(api_key=st.secrets['ANTHROPIC_API_KEY']),
                'voyage': VoyageClient(api_key=st.secrets['VOYAGE_API_KEY']),
                'llama_parse': LlamaParse(api_key=st.secrets['LLAMA_PARSE_API_KEY'])
            }

            # Initialize Qdrant index
            index = QdrantIndex(
                url=config['qdrant_cluster'],
                api_key=st.secrets['QDRANT_API_KEY'],
                index_name=config['index_name']
            )

            urls = get_urls_from_sitemap(st.session_state['sitemap_url'])
            if not urls:
                st.error("No PDF URLs found in the sitemap")
                return

            st.info(f"Found {len(urls)} PDF documents")

            overall_progress = st.progress(0)
            status_text = st.empty()

            num_workers = min(5, len(urls))  # Limit the number of threads
            processed_count = 0

            def update_progress(future):
                nonlocal processed_count
                processed_count += 1
                overall_progress.progress(processed_count / len(urls))
                url = future_to_url[future]
                try:
                    future.result()
                    status_text.text(f"Successfully processed: {url}")
                except Exception as exc:
                    st.error(f"Error processing {url}: {exc}")
                    logger.exception(f"Exception occurred while processing {url}")

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_url = {executor.submit(process_single_document, url, clients, index, config): url for url in urls}
                for future in as_completed(future_to_url):
                    update_progress(future)

            st.success(f"Processing complete! Index name: {config['index_name']}")

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            logger.exception("Exception occurred during processing")

def main():
    render_ui()

if __name__ == "__main__":
    main()


