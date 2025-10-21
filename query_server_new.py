# query_server_dev_xinference-api.py - UNIFIED COLLECTION VERSION
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import NodeWithScore, TextNode
import logging
from typing import List, Tuple, Any, Optional, Dict
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import requests
import socket
import numpy as np
import random
from transformers import AutoTokenizer
from xinference.client import Client
import uvicorn
from contextlib import asynccontextmanager
from log import log, set_request_context, clear_request_context, timed, get_token_length
import asyncio
from dotenv import load_dotenv
from qdrant_client.http import models
import torch
import traceback
import json
from datetime import datetime
import numpy as np
import time
from fastapi.responses import JSONResponse

random.seed(42)

def sigmoid(x):
    """Computes the sigmoid function, mapping a logit to a probability-like score."""
    # Xử lý các giá trị lớn để tránh overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
load_dotenv()

# Constants
EMBED_MODEL_PATH = os.environ.get("EMBED_MODEL_PATH", "./models/bge_m3")
BM25_PERSIST_PATH = os.environ.get("BM25_PERSIST_PATH")
TOKENIZER_MODEL = os.environ.get("TOKENIZER_MODEL")
QDRANT_COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "unified_documents")
QDRANT_DENSE_TOP_K = int(os.environ.get("QDRANT_DENSE_TOP_K", 50))
BM25_TOP_K = int(os.environ.get("BM25_TOP_K", 30))
RERANKER_THRESHOLD = float(os.environ.get("RERANKER_THRESHOLD", 0.3))
MAX_CONTEXT_LENGTH = int(os.environ.get("MAX_CONTEXT_LENGTH", 2048))
PARENT_TEXT_MAX_LENGTH = int(os.environ.get("PARENT_TEXT_MAX_LENGTH", 1024))
MAX_DENSE_RESULTS = int(os.environ.get("MAX_DENSE_RESULTS", 30))
MAX_BM25_RESULTS = int(os.environ.get("MAX_BM25_RESULTS", 15))
EMBED_CHUNK_SIZE = int(os.environ.get("EMBED_CHUNK_SIZE", 1024))
RERANK_MODEL_NAME = os.environ.get("RERANK_MODEL_NAME")
QDRANT_HOST = os.getenv('QDRANT_HOST')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
ROUTE_THRESHOLD = float(os.environ.get("ROUTE_THRESHOLD", 0.3))
LLM_MODEL_NAME = os.environ.get("TOKENIZER_MODEL")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File logging setup
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)

# Pydantic models
class QueryRequest(BaseModel):
    topic: Optional[str] = None   # Optional
    chat_history: str
    user_input: str
    user_id: str
    session_id: str
    transaction_id: str

class BM25Details(BaseModel):
    term_frequencies: Dict[str, int]
    doc_length: int
    avg_doc_length: float
    score: float
    matched_terms: List[str]

class RetrievalDetails(BaseModel):
    context: str
    structured_context: List[Dict[str, Any]] = []
    reranking_scores: List[Dict[str, Any]]
    stats: Dict[str, Any]
    original_dense_texts: List[Dict[str, Any]]
    original_bm25_texts: List[Dict[str, Any]] 

class QueryResponse(BaseModel):
    response: str
    doc_metadata: List[Dict[str, Any]] 
    retrieval_metrics: RetrievalDetails
    doc_ids: str

# Updated: Keep topic mapping for backward compatibility and optional filtering
VALID_TOPICS = {
    "tcns": "tcns",
    "dtpt": "dtpt",
    "qlcl": "qlcl", 
    "ktcn": "ktcn",
    "ncpt": "ncpt",
    "ktpc": "ktpc",
    "vptl": "vptl",
    "temBC": "temBC",
    "ttds": "ttds",
    "dvkh": "dvkh",
    "bd_vhx": "bd_vhx",
    "bccp_nd": "bccp_nd",
    "bccp_qt": "bccp_qt",
    "hcc": "hcc",
    "ppbl": "ppbl",
    "tcbc": "tcbc",

}

# Global variables
embed_model = None
tokenizer = None
client = None
reranker = None
bm25_retriever = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    log("info", "Application startup: Initializing components")
    initialize_components()
    yield
    log("info", "Application shutdown: Cleaning up resources")
    if client:
        client.close()
    if embed_model:
        del embed_model
    if tokenizer:
        del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

app = FastAPI(lifespan=lifespan)

def perform_network_diagnostics():
    log("info", "Performing network diagnostics...")
    
    try:
        ip_address = socket.gethostbyname(QDRANT_HOST)
        log("info", f"DNS resolution successful. {QDRANT_HOST} resolves to {ip_address}")
    except socket.gaierror as e:
        log("error", f"DNS resolution failed for {QDRANT_HOST}: {str(e)}")
        return False

    try:
        with socket.create_connection((QDRANT_HOST, QDRANT_PORT), timeout=5) as sock:
            log("info", f"TCP connection to {QDRANT_HOST}:{QDRANT_PORT} successful")
    except (socket.timeout, ConnectionRefusedError) as e:
        log("error", f"TCP connection to {QDRANT_HOST}:{QDRANT_PORT} failed: {str(e)}")
        return False

    return True

def check_qdrant_health():
    endpoints = ['/collections']
    for endpoint in endpoints:
        try:

            url = f"http://{QDRANT_HOST}:{QDRANT_PORT}{endpoint}"
            log("info", f"Attempting health check with endpoint: {url}")
            response = requests.get(url)
            response.raise_for_status()
            log("info", f"Qdrant health check passed using endpoint: {endpoint}")
            if endpoint == '/collections':
                collections = response.json().get('result', {}).get('collections', [])
                log("info", f"Available collections: {collections}")
            return True
        except requests.RequestException as e:
            log("warning", f"Qdrant health check failed for endpoint {endpoint}: {str(e)}")
    
    log("error", "All health check attempts failed")
    return False

@timed
def initialize_components():
    global embed_model, tokenizer, client, bm25_retriever, reranker

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_PATH,
        embed_batch_size=int(os.environ.get("EMBED_MODEL_BATCH_SIZE", 8)),
        max_length=int(os.environ.get("EMBED_MODEL_MAX_LENGTH", 1024)),
        trust_remote_code=True,
    )
    Settings.embed_model = embed_model

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    # Check Qdrant health
    if not check_qdrant_health():
        raise Exception("Qdrant server is not healthy or not reachable")

    # Initialize Qdrant client
    try:
        log("info", f"Attempting to connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30.0, prefer_grpc=False)
        
        collections = client.get_collections()
        log("info", f"Successfully connected to Qdrant. Available collections: {collections}")
    except UnexpectedResponse as e:
        log("error", f"Failed to connect to Qdrant: Unexpected response: {str(e)}")
        raise
    except Exception as e:
        log("error", f"Failed to connect to Qdrant: {str(e)}")
        raise

    # Changed: Initialize single unified BM25 retriever
    unified_bm25_path = os.path.join(BM25_PERSIST_PATH, "unified")
    if os.path.exists(unified_bm25_path):
        bm25_retriever = BM25Retriever.from_persist_dir(unified_bm25_path)
        bm25_retriever.similarity_top_k = BM25_TOP_K
        logger.info("Unified BM25 retriever initialized successfully")
    else:
        logger.warning(f"Unified BM25 retriever directory not found at {unified_bm25_path}")
        bm25_retriever = None
    
    # Initialize reranker
    reranker = XinferenceReranker()
    reranker.initialize_model()
    log("info", "All components initialized for unified collection")

def is_valid_text(text):
    return text not in ["N/A", "nan", "NaT", ""] and isinstance(text, str)

def get_answer(text: str) -> str:
    lines = text.split("\n***o***\n")
    return str(lines[1]) if len(lines) > 1 else text

def process_tables_info(tables_info: List[str]) -> List[str]:
    processed_tables = []
    for table in tables_info:
        parts = table.split(": ", 1)
        if len(parts) == 2:
            name, url = parts
            if name.lower() != "nan" and url.lower() != "nan":
                processed_tables.append(f"{name}: {url}")
    return processed_tables

def get_chunk_with_context(
    result: NodeWithScore, 
    max_tokens: int = PARENT_TEXT_MAX_LENGTH
) -> Dict[str, Any]:
    """
    Lấy chunk kèm context từ metadata với token limit.
    Window size = 1 (1 chunk trước + chunk hiện tại + 1 chunk sau)
    
    Args:
        result: NodeWithScore object
        max_tokens: Giới hạn token tối đa
        
    Returns:
        Dict chứa full text và metadata
    """
    metadata = result.node.metadata
    content = result.node.text
    
    # Validate context helper
    def validate_context(ctx):
        if not ctx or not isinstance(ctx, str):
            return None
        ctx = ctx.strip()
        if ctx.lower() in ['n/a', 'nan', 'nat', 'none', '']:
            return None
        return ctx
    
    context_before = validate_context(metadata.get('context_before_text'))
    context_after = validate_context(metadata.get('context_after_text'))
    
    # Calculate tokens
    content_tokens = get_token_length(content)
    
    # Nếu content đã quá dài, chỉ trả về content
    if content_tokens >= max_tokens:
        log("warning", f"Content alone exceeds limit: {content_tokens}/{max_tokens} tokens")
        return {
            "text": content,
            "has_context_before": False,
            "has_context_after": False,
            "was_truncated": True,
            "final_tokens": content_tokens,
            "context_before_tokens": 0,
            "context_after_tokens": 0,
            "content_tokens": content_tokens,
            "chunk_index": metadata.get('chunk_index', -1),
            "total_chunks": metadata.get('total_chunks', 0)
        }
    
    available_tokens = max_tokens - content_tokens
    
    # Phân bổ token: 40% cho before, 60% cho after (context sau thường quan trọng hơn)
    max_before_tokens = int(available_tokens * 0.4)
    max_after_tokens = available_tokens - max_before_tokens
    
    parts = []
    actual_before_tokens = 0
    actual_after_tokens = 0
    
    # ===== XỬ LÝ CONTEXT BEFORE =====
    if context_before and max_before_tokens > 50:  # Tối thiểu 50 tokens mới thêm
        # Vì window_size=1, context_before chỉ chứa 1 chunk duy nhất
        # Không cần split bằng separator
        ctx_tokens = get_token_length(context_before)
        
        if ctx_tokens <= max_before_tokens:
            # Context before vừa đủ
            parts.append(f"[Ngữ cảnh trước]\n{context_before}\n")
            actual_before_tokens = ctx_tokens
        else:
            # Cần truncate - lấy phần cuối (gần main content nhất)
            truncate_ratio = max_before_tokens / ctx_tokens
            char_limit = int(len(context_before) * truncate_ratio)
            truncated = "..." + context_before[-char_limit:]
            parts.append(f"[Ngữ cảnh trước]\n{truncated}\n")
            actual_before_tokens = max_before_tokens
            
        log("debug", f"Added context_before: {actual_before_tokens} tokens")
    
    # ===== MAIN CONTENT =====
    parts.append(f"[Nội dung chính]\n{content}\n")
    
    # ===== XỬ LÝ CONTEXT AFTER =====
    if context_after and max_after_tokens > 50:
        ctx_tokens = get_token_length(context_after)
        
        if ctx_tokens <= max_after_tokens:
            # Context after vừa đủ
            parts.append(f"[Ngữ cảnh sau]\n{context_after}")
            actual_after_tokens = ctx_tokens
        else:
            # Cần truncate - lấy phần đầu (gần main content nhất)
            truncate_ratio = max_after_tokens / ctx_tokens
            char_limit = int(len(context_after) * truncate_ratio)
            truncated = context_after[:char_limit] + "..."
            parts.append(f"[Ngữ cảnh sau]\n{truncated}")
            actual_after_tokens = max_after_tokens
            
        log("debug", f"Added context_after: {actual_after_tokens} tokens")
    
    full_text = "\n".join(parts)
    final_tokens = get_token_length(full_text)
    
    return {
        "text": full_text,
        "has_context_before": context_before is not None,
        "has_context_after": context_after is not None,
        "was_truncated": final_tokens >= max_tokens * 0.95,  # Coi như truncated nếu dùng >95% limit
        "final_tokens": final_tokens,
        "context_before_tokens": actual_before_tokens,
        "context_after_tokens": actual_after_tokens,
        "content_tokens": content_tokens,
        "chunk_index": metadata.get('chunk_index', -1),
        "total_chunks": metadata.get('total_chunks', 0)
    }

class XinferenceReranker:
    def __init__(self, xinference_endpoint=os.environ.get("XINFERENCE_ENDPOINT")):
        self.xinference_client = Client(xinference_endpoint)
        self.rerank_model = None

    def initialize_model(self):
        if self.rerank_model is None:
            print("Connecting to the reranker model...")
            try:
                models = self.xinference_client.list_models()

                for model_id, model_info in models.items():
                    if (
                        model_info["model_name"] == RERANK_MODEL_NAME
                        and model_info["model_type"] == "rerank"
                    ):
                        self.rerank_model = self.xinference_client.get_model(model_id)
                        print(f"Connected to reranker model. Model ID: {model_id}")
                        return

                if not self.rerank_model:
                    print("Custom reranker model not found among launched models.")
                    print("Available models:")
                    print(models)
            except Exception as e:
                print(f"Error connecting to reranker model: {e}")
        else:
            print("Reranker model already initialized.")

    def rerank_results(self, all_results, query, topic, max_context_length, max_results, tokenizer, RERANKER_THRESHOLD):
        """
        Rerank results with improved error handling and validation
        """
        # FIX 1: Check if input is empty
        if not all_results:
            log("warning", "Reranking skipped: Input list is empty.")
            return []
        
        if self.rerank_model is None:
            self.initialize_model()

        log("info", f"Reranking {len(all_results)} candidates for query: '{query[:50]}...'")

        # FIX 2: Validate and clean results BEFORE creating corpus
        valid_results = []
        for i, res in enumerate(all_results):
            if not hasattr(res, 'node'):
                log("warning", f"Skipping result {i}: Missing 'node' attribute")
                continue
            if not hasattr(res.node, 'text'):
                log("warning", f"Skipping result {i}: Missing 'text' attribute in node")
                continue
            if not res.node.text or not res.node.text.strip():
                log("warning", f"Skipping result {i}: Empty or whitespace-only text")
                continue
            valid_results.append(res)
        
        if not valid_results:
            log("warning", "Reranking skipped: No valid text candidates after cleaning.")
            return []
        
        log("info", f"Valid candidates after cleaning: {len(valid_results)}/{len(all_results)}")
        
        # FIX 3: Create corpus from validated results only
        corpus = [result.node.text for result in valid_results]
        
        # FIX 4: Additional validation before sending to reranker
        if not corpus or len(corpus) == 0:
            log("error", "Corpus is empty after validation!")
            return []
        
        # Log corpus preview
        log("debug", f"Corpus preview (first 3 items): {[c[:100] for c in corpus[:3]]}")
        
        try:
            # FIX 5: Wrap reranker call in try-except
            rerank_result = self.rerank_model.rerank(corpus, query)
            
            # FIX 6: Validate reranker response
            if not rerank_result or "results" not in rerank_result:
                log("error", f"Invalid reranker response: {rerank_result}")
                return []
            
            if not rerank_result["results"]:
                log("warning", "Reranker returned empty results")
                return []
            
            log("debug", f"Reranker returned {len(rerank_result['results'])} scores")
            
        except Exception as e:
            log("error", f"Reranker API call failed: {str(e)}")
            log("debug", f"Corpus length: {len(corpus)}, Query: '{query[:100]}'")
            return []

        # FIX 7: Validate indices in reranker results
        results_with_scores = []
        for item in rerank_result["results"]:
            idx = item.get("index")
            if idx is None:
                log("warning", "Reranker result missing 'index' field")
                continue
            if idx < 0 or idx >= len(valid_results):
                log("warning", f"Reranker returned invalid index {idx} (corpus size: {len(valid_results)})")
                continue
            
            score = item.get("relevance_score", 0)
            results_with_scores.append((
                valid_results[idx],
                score,
                sigmoid(score)
            ))

        if not results_with_scores:
            log("warning", "No valid results after processing reranker output")
            return []

        # Log score distribution
        all_scores = [score for _, score, _ in results_with_scores]
        log("info", f"Reranker scores - min: {min(all_scores):.3f}, max: {max(all_scores):.3f}, avg: {sum(all_scores)/len(all_scores):.3f}")

        # Filter by threshold (using logit scores)
        threshold = RERANKER_THRESHOLD
        filtered_sorted_results = sorted(
            [result for result in results_with_scores if result[1] >= threshold],
            key=lambda x: x[1],
            reverse=True,
        )

        log("info", f"Results after threshold filtering ({threshold}): {len(filtered_sorted_results)}/{len(results_with_scores)}")

        # Trim by token limit
        top_results = []
        total_token_count = 0
        
        for result, score, prob in filtered_sorted_results:
            result_tokens = get_token_length(result.node.text)
            if (total_token_count + result_tokens > max_context_length or 
                len(top_results) >= max_results):
                break
            top_results.append((result, score, prob))
            total_token_count += result_tokens

        log("info", f"Final reranked results: {len(top_results)} (total tokens: {total_token_count})")

        # Reverse to put best results last (if needed for your logic)
        top_results.reverse()

        # Fallback: if nothing passed, take the best one
        if not top_results and filtered_sorted_results:
            log("warning", "No results fit token limit, taking best single result")
            top_results = [filtered_sorted_results[0]]

        return top_results

def get_embeddings(texts):
    embeddings = embed_model.get_text_embedding(texts)
    return embeddings

# Changed: Updated to use unified collection without topic-specific naming
# Thay thế hàm cũ bằng hàm đã sửa lỗi này
def multi_vector_retrieve(query: str, topic_filter: str = None):
    logging.debug(f"Starting multi_vector_retrieve for query: {query}")
    if topic_filter:
        logging.debug(f"With topic filter: {topic_filter}")

    # Generate vectors for the query
    embeddings_list = {}
    for chunk_size in [128, 256, 512]:
        Settings.chunk_size = chunk_size
        embeddings = get_embeddings(query)
        embeddings_list[f"text-dense-{chunk_size}"] = embeddings

    try:
        # Build search request
        search_params = {
            "collection_name": QDRANT_COLLECTION_NAME,  # Use unified collection
            "prefetch": [
                models.Prefetch(
                    query=embeddings_list["text-dense-128"],
                    using="text-dense-128",
                    limit=15,
                ),
                models.Prefetch(
                    query=embeddings_list["text-dense-256"],
                    using="text-dense-256",
                    limit=15,
                ),
                models.Prefetch(
                    query=embeddings_list["text-dense-512"],
                    using="text-dense-512",
                    limit=15,
                ),
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "score_threshold": 0.3,
            "limit": QDRANT_DENSE_TOP_K
        }

        # Add topic filter if specified
        if topic_filter and topic_filter in VALID_TOPICS:
            # SỬA LỖI Ở ĐÂY: đổi "filter" thành "query_filter"
            search_params["query_filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="topic",
                        match=models.MatchValue(value=topic_filter)
                    )
                ]
            )
            logging.debug(f"Applied topic filter for: {topic_filter}")

        results = client.query_points(**search_params)
        logging.debug(f"Query completed. Number of results: {len(results.points)}")
    except Exception as e:
        logging.error(f"Error in query_points: {str(e)}")
        raise

    return results

def generate_vector(query):
    logging.debug(f"Generating vector for query: {query}")
    try:
        vector = embed_model.get_text_embedding(query)
        if isinstance(vector, list):
            vector = np.array(vector)
        logging.debug(f"Vector generated. Shape: {vector.shape}")
        return vector.tolist()
    except Exception as e:
        logging.error(f"Error in generate_vector: {str(e)}")
        raise

def log_result(result, retrieval_type):
    return {
        "retrieval_type": retrieval_type,
        "doc_id": result.payload.get("doc_id", "Unknown") if retrieval_type == "dense" else result.metadata.get("doc_id", "Unknown"),
        "score": result.score,
        "text": result.payload.get("content", "")[:100] if retrieval_type == "dense" else result.text[:100],
        "metadata": {k: v for k, v in (result.payload if retrieval_type == "dense" else result.metadata).items() if k != "content"}
    }

def get_bm25_details(bm25_retriever, query: str, result: NodeWithScore) -> BM25Details:
    doc_text = result.text
    terms = query.lower().split()
    term_freqs = {}
    
    for term in terms:
        term_freqs[term] = doc_text.lower().count(term)
    
    doc_length = len(doc_text.split())
    avg_doc_length = bm25_retriever.avg_doc_length if hasattr(bm25_retriever, 'avg_doc_length') else 0.0
    matched_terms = [term for term in terms if term_freqs[term] > 0]
    
    return BM25Details(
        term_frequencies=term_freqs,
        doc_length=doc_length,
        avg_doc_length=avg_doc_length,
        score=result.score,
        matched_terms=matched_terms
    )

def parse_date_flexible(date_str):
    if not isinstance(date_str, str) or date_str in ["N/A", "nan", "NaT", ""]:
        return None
    
    date_formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None

@timed
async def retrieve_and_process(
    query: str, 
    topic_filter: str = None,  # Changed: topic is now optional filter
    user_id: str = "",
    session_id: str = "",
    transaction_id: str = "",
    max_context_length: int = MAX_CONTEXT_LENGTH,
    parent_text_max_length: int = PARENT_TEXT_MAX_LENGTH
) -> Tuple[str, List[Dict[str, Any]], RetrievalDetails, str]:

    log("debug", "Starting unified retrieval and processing", 
        topic_filter=topic_filter, query_length=get_token_length(query))

    # Perform multi-vector retrieval with optional topic filter
    dense_task = asyncio.create_task(asyncio.to_thread(multi_vector_retrieve, query, topic_filter))
    dense_results = await dense_task
    log("info", f"Dense retrieval found {len(dense_results.points)} results.")
    dense_results_log = [log_result(r, "dense") for r in dense_results.points]
    log("info", "All dense retrieval results", results=json.dumps(dense_results_log, indent=2, ensure_ascii=False))

    # Process dense results
    original_dense_texts = []
    for point in dense_results.points:
        original_dense_texts.append({
            "doc_id": point.payload.get("doc_id", "Unknown"),
            "text": point.payload.get("content", ""),
            "score": point.score,
            "topic": point.payload.get("topic", "Unknown"),  # Include topic in results
            "metadata": {
                "doc_title": point.payload.get("doc_title", "N/A"),
                "doc_date": point.payload.get("doc_date", "N/A"),
                "file_type": point.payload.get("file_type", "N/A"),
                "chunk_size": point.payload.get("chunk_size", "N/A")
            }
        })

    # Process BM25 results with optional topic filtering
    bm25_results = []
    original_bm25_texts = []
    if bm25_retriever is not None:
        bm25_task = asyncio.create_task(asyncio.to_thread(bm25_retriever.retrieve, query))
        bm25_results = await bm25_task
        log("info", f"BM25 retrieval found {len(bm25_results)} results.")
        if topic_filter and topic_filter in VALID_TOPICS:
            bm25_results = [r for r in bm25_results if r.metadata.get('topic') == topic_filter]
            log("info", f"Filtered BM25 results by topic {topic_filter}: {len(bm25_results)} results")
        
        bm25_details = []
        for result in bm25_results:
            bm25_details.append(get_bm25_details(bm25_retriever, query, result))
            original_bm25_texts.append({
                "doc_id": result.metadata.get("doc_id", "Unknown"),
                "text": result.text,
                "score": result.score,
                "topic": result.metadata.get("topic", "Unknown"),  # Include topic
                "metadata": {
                    "doc_title": result.metadata.get("doc_title", "N/A"),
                    "doc_date": result.metadata.get("doc_date", "N/A"),
                    "file_type": result.metadata.get("file_type", "N/A")
                },
                "term_frequencies": {
                    term: result.text.lower().count(term)
                    for term in query.lower().split()
                }
            })

        bm25_results_log = [
            {
                "doc_id": result.metadata.get("doc_id", "Unknown"),
                "score": result.score,
                "topic": result.metadata.get("topic", "Unknown"),
                "matched_terms": details.matched_terms,
                "term_frequencies": details.term_frequencies,
                "doc_length": details.doc_length
            }
            for result, details in zip(bm25_results, bm25_details)
        ]
        log("info", "All BM25 retrieval results",
            results=json.dumps(bm25_results_log, indent=2, ensure_ascii=False))
    else:
        bm25_results = []
        log("warning", "BM25 retriever not available")

    # Rest of the processing remains the same...
    vector_results = [NodeWithScore(node=TextNode(text=r.payload.get("content", ""), metadata=r.payload), score=r.score) for r in dense_results.points]
    processed_bm25_results = [NodeWithScore(node=TextNode(text=r.text, metadata=r.metadata), score=r.score) for r in bm25_results]

    # --- BẮT ĐẦU THAY ĐỔI LOGIC RETRIEVER/RERANKER ---

    # 1. Rerank các kết quả từ Dense search
    log("info", f"Reranking {len(vector_results)} dense candidates...")
    if not vector_results:
        log("warning", "No dense results to rerank")
        reranked_dense = []
    else:
        log("info", f"Reranking {len(vector_results)} dense candidates...")
        reranked_dense = reranker.rerank_results(
            vector_results, query, topic_filter or "unified",
            max_context_length, MAX_DENSE_RESULTS, tokenizer, RERANKER_THRESHOLD
        )
    log("info", f"Dense reranking completed: {len(reranked_dense)} results")

    # 2. Rerank các kết quả từ BM25
    log("info", f"Reranking {len(processed_bm25_results)} BM25 candidates...")
    if not processed_bm25_results:
        log("warning", "No BM25 results to rerank")
        reranked_bm25 = []
    else:
        log("info", f"Reranking {len(processed_bm25_results)} BM25 candidates...")
        reranked_bm25 = reranker.rerank_results(
            processed_bm25_results, query, topic_filter or "unified",
            max_context_length, MAX_BM25_RESULTS, tokenizer, RERANKER_THRESHOLD
        )
    log("info", f"BM25 reranking completed: {len(reranked_bm25)} results")

    # 3. Gộp và deduplicate
    all_reranked = reranked_dense + reranked_bm25
    log("info", f"Total before deduplication: {len(all_reranked)}")

    unique_results = {}
    for result, logit, prob in all_reranked:
        text_hash = hash(result.node.text)
        if text_hash not in unique_results:
            unique_results[text_hash] = (result, logit, prob, "first")
        else:
            # Giữ score cao hơn và track source
            if logit > unique_results[text_hash][1]:
                unique_results[text_hash] = (result, logit, prob, "replaced")

    duplicates_removed = len(all_reranked) - len(unique_results)
    if duplicates_removed > 0:
        log("info", f"Removed {duplicates_removed} duplicate results")

    # 4. Sort theo reranker logit
    final_results = [(r, l, p) for r, l, p, _ in unique_results.values()]
    final_results.sort(key=lambda x: x[1], reverse=True)
    log("info", f"Sorted {len(final_results)} unique results by reranker score")

    # 5. Trim theo token limit
    final_results_trimmed = []
    total_tokens = 0
    MAX_FINAL_TOKENS = int(os.environ.get("MAX_FINAL_CONTEXT_TOKENS", MAX_CONTEXT_LENGTH))

    for result, logit, prob in final_results:
        result_tokens = get_token_length(result.node.text)
        if total_tokens + result_tokens > MAX_FINAL_TOKENS:
            log("debug", f"Stopping at {len(final_results_trimmed)} results due to token limit")
            break
        final_results_trimmed.append((result, logit, prob))
        total_tokens += result_tokens

    final_results = final_results_trimmed
    log("info", f"Final results after token trimming: {len(final_results)} "
            f"({total_tokens}/{MAX_FINAL_TOKENS} tokens)")

    # 6. Tạo reranking_details với source tracking chính xác
    reranking_details = []
    reranked_dense_texts = {res[0].node.text for res in reranked_dense}
    reranked_bm25_texts = {res[0].node.text for res in reranked_bm25}

    for result, logit, prob in final_results:
        # Determine source (có thể có trong cả 2, ưu tiên dense nếu trùng)
        if result.node.text in reranked_dense_texts:
            source = "dense_reranked"
        elif result.node.text in reranked_bm25_texts:
            source = "bm25_reranked"
        else:
            source = "unknown"
        
        reranking_details.append({
            "doc_id": result.node.metadata.get("doc_id", "Unknown"),
            "doc_date": result.node.metadata.get("doc_date", "N/A"),
            "original_score": result.score,
            "final_logit": logit,
            "final_probability": prob,
            "source": source,
            "topic": result.node.metadata.get("topic", "Unknown"),
            "text_preview": result.node.text[:200],
        })

    # 1. Xây dựng Context CHO LLM với full context
    # XÂY DỰNG CONTEXT CHO LLM với full context
    context_text_list = []
    structured_context_list = []
    doc_counter = Counter()
    source_info_to_metadata_map = {}
    
    for node_with_score, final_logit, final_prob in final_results:
        metadata_info = node_with_score.node.metadata
        doc_id = metadata_info.get("doc_id", "N/A")
        doc_date = metadata_info.get("doc_date", "N/A")
        doc_title = metadata_info.get("doc_title", "N/A")

        # LẤY CHUNK KÈM CONTEXT
        chunk_with_context = get_chunk_with_context(
            node_with_score, 
            max_tokens=parent_text_max_length
        )
        # Lấy full context
        full_text_with_context = chunk_with_context["text"]
        # Log nếu context bị truncate
        if chunk_with_context["was_truncated"]:
            log("debug", 
                f"Chunk {chunk_with_context['chunk_index']}/{chunk_with_context['total_chunks']} "
                f"truncated: {chunk_with_context['final_tokens']} tokens "
                f"(content: {chunk_with_context['content_tokens']}, "
                f"before: {chunk_with_context['context_before_tokens']}, "
                f"after: {chunk_with_context['context_after_tokens']})")
            
        # Kiểm tra token limit
        content_tokens = get_token_length(full_text_with_context)
        if content_tokens > parent_text_max_length:
            # Nếu quá dài, fallback về chỉ dùng content chính
            log("debug", f"Context too long ({content_tokens} tokens), using content only")
            full_text_with_context = node_with_score.node.text

        structured_context_list.append({
            "doc_id": doc_id,
            "doc_date": doc_date,
            "doc_title": doc_title,
            "text": full_text_with_context,
            "final_logit": final_logit,
            "final_probability": final_prob,
            "has_context_before": chunk_with_context["has_context_before"],
            "has_context_after": chunk_with_context["has_context_after"],
            "was_truncated": chunk_with_context["was_truncated"],
            "chunk_index": chunk_with_context["chunk_index"],
            "total_chunks": chunk_with_context["total_chunks"],
            "final_tokens": chunk_with_context["final_tokens"],
            "context_tokens": {
                "before": chunk_with_context["context_before_tokens"],
                "after": chunk_with_context["context_after_tokens"],
                "content": chunk_with_context["content_tokens"]
            },
            "metadata": {k: v for k, v in metadata_info.items() 
                        if k not in ["content", "context_before_text", "context_after_text"]}
        })
    
        
        # Build source info (giữ nguyên code cũ)
        source_info = " - ".join(filter(None, [
            f"Văn bản số {doc_id}" if doc_id not in ["N/A", "nan", "NaT", "website"] else None,
            f"ngày {parse_date_flexible(metadata_info.get('doc_date', 'N/A')).strftime('%d-%m-%Y')}" if parse_date_flexible(metadata_info.get('doc_date', 'N/A')) else None,
            f"v/v {doc_title}" if doc_title not in ["N/A", "nan", "NaT", "website"] else doc_title
        ]))

        if source_info:
            doc_counter.update([source_info])
            if source_info not in source_info_to_metadata_map:
                source_info_to_metadata_map[source_info] = {
                    "doc_id": doc_id, 
                    "doc_date": doc_date, 
                    "doc_title": doc_title
                }

    context_text = "\n***\n".join([item["text"] for item in structured_context_list])

    # Logic tạo final_references_structured (doc_metadata) và doc_ids_string
    final_references_structured = []
    unique_doc_ids_for_response = set()

    # Thêm tài liệu đầu tiên (có điểm cao nhất) vào danh sách tham khảo
    if final_results:
        top_result_node = final_results[0][0].node # Lấy node từ tuple (node, score, prob)
        top_metadata = top_result_node.metadata
        top_doc_id = top_metadata.get('doc_id', 'N/A')
        top_doc_date = top_metadata.get('doc_date', 'N/A')
        top_doc_title = top_metadata.get('doc_title', 'N/A')
        top_source_info = " - ".join(filter(None, [
            f"Văn bản số {top_doc_id}" if top_doc_id not in ["N/A", "nan", "NaT", "website"] else None,
            f"ngày {parse_date_flexible(top_metadata.get('doc_date', 'N/A')).strftime('%d-%m-%Y')}" if parse_date_flexible(top_metadata.get('doc_date', 'N/A')) else None,
            f"v/v {top_metadata.get('doc_title')}" if top_metadata.get('doc_title') not in ["N/A", "nan", "NaT", "website"] else top_metadata.get('doc_title')
        ]))
        if top_source_info:
            final_references_structured.append({
                "display_text": top_source_info, 
                "doc_id": top_doc_id,
                "doc_date": top_doc_date,
                "doc_title": top_doc_title
            })
            unique_doc_ids_for_response.add(top_doc_id)

    # Thêm các nguồn được tham chiếu phổ biến nhất (từ doc_counter)
    # Đảm bảo không trùng lặp và giới hạn số lượng
    MAX_REFERENCE_DOCS=10
    most_common_sources = doc_counter.most_common()
    for source, count in most_common_sources:
        if len(final_references_structured) >= MAX_REFERENCE_DOCS: break # Giới hạn số lượng doc_metadata
        source_doc_id = source_info_to_metadata_map.get(source, {}).get("doc_id")
        if source_doc_id and source_doc_id not in unique_doc_ids_for_response:
            # Lấy đầy đủ thông tin metadata từ map đã tạo trước đó
            source_metadata = source_info_to_metadata_map.get(source, {})
            final_references_structured.append({
                "display_text": source, 
                "doc_id": source_doc_id,
                "doc_date": source_metadata.get("doc_date", "N/A"), 
                "doc_title": source_metadata.get("doc_title", "N/A")
            })
            unique_doc_ids_for_response.add(source_doc_id)
            
    # Tạo chuỗi doc_ids từ tất cả các doc_id có trong final_references_structured
    doc_ids_string = ",".join(sorted(list(unique_doc_ids_for_response)))
    
    # Cập nhật retrieval_stats
    retrieval_stats = {
        "total_final_results": len(final_results),
        "dense_reranked_results": len(reranked_dense),
        "bm25_reranked_results": len(reranked_bm25),  # Đổi tên key
        "total_candidates_before_dedup": len(all_reranked),
        "duplicates_removed": duplicates_removed,
        "total_dense_candidates_initial": len(vector_results),
        "total_bm25_candidates_initial": len(processed_bm25_results),
        "topic_filter": topic_filter,
        "reranker_threshold": RERANKER_THRESHOLD,
        "average_scores": {
            "dense_original_avg": sum(r.score for r in dense_results.points) / len(dense_results.points) if dense_results.points else 0,
            "bm25_original_avg": sum(r.score for r in bm25_results) / len(bm25_results) if bm25_results else 0,
            "final_reranked_avg_logit": sum(logit for _, logit, _ in final_results) / len(final_results) if final_results else 0,
            "final_reranked_avg_prob": sum(prob for _, _, prob in final_results) / len(final_results) if final_results else 0,
        },
        "query_info": {
            "query_length_tokens": get_token_length(query),
            "query_terms": query.lower().split()
        },
        "context_info": {
            "chunks_with_context_before": sum(
                1 for item in structured_context_list 
                if item.get('has_context_before', False)
            ),
            "chunks_with_context_after": sum(
                1 for item in structured_context_list 
                if item.get('has_context_after', False)
            ),
            "total_chunks": len(structured_context_list)
        }
    }

    # ===== LOG CONTEXT STATISTICS =====
    context_usage_stats = {
        "total_chunks": len(structured_context_list),
        "with_both_contexts": sum(
            1 for item in structured_context_list 
            if item["has_context_before"] and item["has_context_after"]
        ),
        "with_before_only": sum(
            1 for item in structured_context_list 
            if item["has_context_before"] and not item["has_context_after"]
        ),
        "with_after_only": sum(
            1 for item in structured_context_list 
            if not item["has_context_before"] and item["has_context_after"]
        ),
        "without_context": sum(
            1 for item in structured_context_list 
            if not item["has_context_before"] and not item["has_context_after"]
        ),
        "truncated_chunks": sum(
            1 for item in structured_context_list 
            if item.get("was_truncated", False)
        ),
        "avg_tokens_per_chunk": sum(
            item["final_tokens"] for item in structured_context_list
        ) / len(structured_context_list) if structured_context_list else 0,
        "avg_context_before_tokens": sum(
            item["context_tokens"]["before"] for item in structured_context_list
        ) / len(structured_context_list) if structured_context_list else 0,
        "avg_context_after_tokens": sum(
            item["context_tokens"]["after"] for item in structured_context_list
        ) / len(structured_context_list) if structured_context_list else 0,
    }

    log("info", "Context usage statistics", stats=context_usage_stats)

    # Thêm vào retrieval_stats
    retrieval_stats["context_usage"] = context_usage_stats

    # 4. Tạo đối tượng RetrievalDetails
    retrieval_details = RetrievalDetails(
        reranking_scores=reranking_details,
        stats=retrieval_stats,
        context=context_text, 
        structured_context=structured_context_list,
        original_dense_texts=original_dense_texts,
        original_bm25_texts=original_bm25_texts,
    )

    return context_text, final_references_structured, retrieval_details, doc_ids_string

@app.post("/rag_query", response_model=QueryResponse)
@timed
async def rag_query_endpoint(request: QueryRequest):
    set_request_context(request.transaction_id, request.user_id, request.session_id)
    
    empty_retrieval_metrics = RetrievalDetails(
        reranking_scores=[],
        stats={"total_final_results": 0, "dense_reranked_results": 0, "bm25_top_results": 0, 
               "total_dense_candidates_initial": 0, "total_bm25_candidates_initial": 0,
               "topic_filter": None,
               "average_scores": {"dense_original": 0, "bm25_original": 0, "final_combined_avg_logit": 0},
               "query_info": {"query_length_tokens": 0, "query_terms": []}},
        context="", 
        structured_context=[], # Đảm bảo structured_context được khởi tạo
        original_dense_texts=[], 
        original_bm25_texts=[]
    )

    try:
        if not request.user_input.strip():
            return QueryResponse(
                response="EMPTY_QUERY",
                doc_metadata=[], 
                retrieval_metrics=empty_retrieval_metrics,
                doc_ids=""
            )

        user_query = request.user_input.replace("?", "").strip()
        
        # Changed: Topic is now optional - use for filtering if valid, otherwise search all
        topic_filter = None
        # Chỉ xử lý nếu request.topic là một chuỗi có nội dung
        if isinstance(request.topic, str) and request.topic.strip():
            if request.topic in VALID_TOPICS:
                topic_filter = request.topic
                log("info", f"Using valid topic filter: {topic_filter}")
            else:
                # Ghi log rõ ràng khi nhận topic không hợp lệ nhưng vẫn tiếp tục tìm kiếm tất cả
                log("warning", f"Received unknown topic '{request.topic}'. Searching across all topics.")
        else:
            # Xử lý cả trường hợp topic là None, chuỗi rỗng "", hoặc chuỗi chỉ có khoảng trắng
            log("info", "No topic provided or topic is empty. Searching across all topics.")
        
        try:
            context_text, doc_metadata, retrieval_details, doc_ids = await retrieve_and_process(
                user_query,
                topic_filter=topic_filter,  # Pass topic filter instead of required topic
                user_id=request.user_id,
                session_id=request.session_id,
                transaction_id=request.transaction_id,
            )
            
            return QueryResponse(
                response=context_text,
                doc_metadata=doc_metadata,
                retrieval_metrics=retrieval_details,
                doc_ids=doc_ids
            )
            
        except ValueError as ve:
            log("error", "Value error in processing", error=str(ve))
            return QueryResponse(
                response="PROCESSING_ERROR",
                doc_metadata=[],
                retrieval_metrics=empty_retrieval_metrics,
                doc_ids=""
            )
            
    except Exception as e:
        log("error", "Unexpected error in rag_query", error=str(e), traceback=traceback.format_exc())
        return QueryResponse(
            response="SERVER_ERROR",
            doc_metadata=[],
            retrieval_metrics=empty_retrieval_metrics,
            doc_ids=""
        )
        
    finally:
        clear_request_context()

@app.post("/debug/chunk_context")
async def debug_chunk_context(doc_id: str, chunk_index: int):
    """
    Debug endpoint để xem context của một chunk cụ thể.
    """
    try:
        # Query Qdrant
        results = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id)
                    ),
                    models.FieldCondition(
                        key="chunk_index",
                        match=models.MatchValue(value=chunk_index)
                    )
                ]
            ),
            limit=1
        )
        
        if not results[0]:
            return {"error": "Chunk not found"}
        
        point = results[0][0]
        
        return {
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "content": point.payload.get("content", "N/A"),
            "context_before": point.payload.get("context_before_text", "N/A"),
            "context_after": point.payload.get("context_after_text", "N/A"),
            "total_chunks": point.payload.get("total_chunks", 0)
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/sample_chunks")
async def debug_sample_chunks(limit: int = 5):
    """Debug endpoint to check chunk data quality"""
    try:
        results = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=limit
        )
        
        samples = []
        for point in results[0]:
            samples.append({
                "id": point.id,
                "has_content": "content" in point.payload,
                "content_length": len(point.payload.get("content", "")),
                "content_preview": point.payload.get("content", "")[:200],
                "doc_id": point.payload.get("doc_id", "N/A"),
                "metadata_keys": list(point.payload.keys())
            })
        
        return {
            "total_samples": len(samples),
            "samples": samples
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    try:
        components_status = {
                "embed_model": embed_model is not None,
                "bm25_retriever": bm25_retriever is not None,
                "reranker": reranker is not None and reranker.rerank_model is not None,
                "qdrant_client": client is not None
                }
        all_healthy = all(components_status.values())

        return {
                "status": "healthy" if all_healthy else "degraded",
                "timestamp":time.time(),
                "components": components_status
                }
    except Exception as e:
        return JSONResponse(
                status_code=503,
                content={"status":"Unhealthy", "error": str(e)}
                )

def main():
    log("info", "Starting initialization process for unified collection")
    try:
        initialize_components()
        log("info", "Initialization completed successfully")
    except Exception as e:
        log("error", f"Initialization failed: {str(e)}")
        log("error", "Full error traceback:", exc_info=True)
        return

    log("info", "Starting the server")
    uvicorn.run(app, host="127.0.0.1", port=8338)

if __name__ == "__main__":
    main()
