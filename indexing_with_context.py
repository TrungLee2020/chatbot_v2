"""
Indexing script with pre-computed context from metadata
NO pipeline splitting - files are already split
"""
import os
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
import pandas as pd
from qdrant_client import QdrantClient
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import Node
from qdrant_client.http import models
import sys
import urllib3
import uuid
from datetime import date

today = date.today()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indexing_full.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv(override=True)

# Constants
DOC_DIR_PATH = os.getenv('DOC_DIR_PATH')
DOC_METADATA_PATH = os.getenv('DOC_METADATA_PATH')
INGESTION_CACHE_PATH = os.getenv('INGESTION_CACHE_PATH')
QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'unified_documents_new')
BM25_PERSIST_PATH = os.getenv('BM25_PERSIST_PATH')
EMBEDDING_MODEL_PATH = os.getenv('EMBEDDING_MODEL_PATH')
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION'))

logger.info(f"EMBEDDING_MODEL_PATH: {EMBEDDING_MODEL_PATH}")
logger.info(f"DOC_DIR_PATH: {DOC_DIR_PATH}")
logger.info(f"QDRANT_COLLECTION_NAME: {QDRANT_COLLECTION_NAME}")


def setup_embedding_model() -> HuggingFaceEmbedding:
    """Set up and return the HuggingFace embedding model."""
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_PATH,
        embed_batch_size=32,
        max_length=512,
        cache_folder="/home/dmst_ai/trunglx/models",
        trust_remote_code=True
    )


def load_metadata(path: str) -> pd.DataFrame:
    """Load metadata from CSV file."""
    try:
        df = pd.read_csv(path)
        df = df.astype(str)
        
        logger.info(f"Loaded {len(df)} metadata rows")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        if 'filename' not in df.columns:
            raise ValueError("Column 'filename' not found in metadata CSV")
        
        # Check for context columns
        has_context_before = 'context_before_text' in df.columns
        has_context_after = 'context_after_text' in df.columns
        
        logger.info(f"Context columns present: before={has_context_before}, after={has_context_after}")
        
        df.set_index('filename', inplace=True)
        return df
        
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        raise


def file_metadata(full_path: str, metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """Get metadata for a file with multiple matching strategies."""
    filename_with_extension = os.path.basename(full_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    
    logger.debug(f"Processing file: {full_path}")

    metadata = {'file_path': full_path}

    # Try multiple matching strategies
    search_keys = [
        filename_with_extension,
        filename_without_extension,
        os.path.basename(full_path),
        full_path,
    ]

    for key in search_keys:
        try:
            if key in metadata_df.index:
                file_meta = metadata_df.loc[key].to_dict()
                metadata.update(file_meta)
                logger.debug(f"Found metadata for {filename_with_extension} using key: {key}")
                return metadata
        except KeyError:
            continue

    # No metadata found - use defaults
    logger.warning(f"No metadata found for file: {filename_with_extension}")
    metadata.update({
        'topic': 'unknown',
        'filename': filename_without_extension,
        'file_name': filename_with_extension,
        'file_type': os.path.splitext(filename_with_extension)[1],
        'context_before_text': 'N/A',
        'context_after_text': 'N/A'
    })

    return metadata


def verify_directory_structure(doc_dir: str) -> int:
    """Verify and log directory structure."""
    logger.info(f"Verifying directory structure for: {doc_dir}")
    if not os.path.exists(doc_dir):
        logger.error(f"Directory does not exist: {doc_dir}")
        raise FileNotFoundError(f"Directory does not exist: {doc_dir}")
    
    total_files = 0
    for root, dirs, files in os.walk(doc_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in ['.txt']:
                total_files += 1
    
    logger.info(f"Total supported files found: {total_files}")
    return total_files


def load_documents(doc_dir: str, metadata_df: pd.DataFrame) -> List[Any]:
    """
    Load documents WITHOUT additional splitting.
    Files are already pre-split chunks with context in metadata.
    """
    total_files = verify_directory_structure(doc_dir)

    try:
        reader = SimpleDirectoryReader(
            doc_dir,
            file_metadata=lambda x: file_metadata(x, metadata_df),
            recursive=True,
            required_exts=['.txt']
        )
        
        logger.info("Loading documents...")
        documents = reader.load_data()
        
        logger.info(f"Loaded {len(documents)}/{total_files} documents")
        
        if not documents:
            logger.warning("No documents loaded")
            return []
        
        # ====================================================================
        # CRITICAL: EXCLUDE METADATA FROM EMBEDDINGS
        # ====================================================================
        metadata_keys_to_exclude = [
            # File identifiers
            'filename',
            'file_path',
            'file_name',
            'file_type',
            
            # Document metadata
            'topic',
            'parent_content',
            'is_used',
            'file_id',
            'tables_name',
            'tables_url',
            
            # Context fields - MOST IMPORTANT
            'context_before_text',
            'context_after_text',
            'context_before_indices',
            'context_after_indices',
            'chunk_index',
            'total_chunks',
            'original_filename'
        ]
        
        for doc in documents:
            doc.excluded_embed_metadata_keys = metadata_keys_to_exclude
            doc.excluded_llm_metadata_keys = metadata_keys_to_exclude
        
        logger.info(f"Applied exclusions to {len(documents)} documents")
        
        return documents
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise


def setup_ingestion_pipeline(cache_path: str) -> IngestionPipeline:
    """
    Set up ingestion pipeline WITHOUT splitting.
    Files are already split, we just need to process them.
    """
    try:
        cached_hashes = IngestionCache.from_persist_path(cache_path)
        logger.info("Cache file found. Running using cache...")
    except:
        cached_hashes = None
        logger.info("No cache file found. Running without cache...")
    
    # NO transformations - files are already split!
    transformations = []
    
    return IngestionPipeline(transformations=transformations, cache=cached_hashes)


def create_text_splitters() -> Dict[str, SentenceSplitter]:
    """Create text splitters for multi-vector embedding."""
    return {
        "128": SentenceSplitter(chunk_size=128, chunk_overlap=64),
        "256": SentenceSplitter(chunk_size=256, chunk_overlap=128),
        "512": SentenceSplitter(chunk_size=512, chunk_overlap=256),
    }


def create_and_populate_multi_vector_index(nodes: List[Node], client: QdrantClient) -> str:
    """
    Create and populate multi-vector index with STREAMING approach.
    Balance between RAM usage and upload efficiency.
    """
    collection_name = QDRANT_COLLECTION_NAME

    try:
        # Check if collection exists
        if client.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' already exists")
            return None

        # ====================================================================
        # CREATE COLLECTION
        # ====================================================================
        logger.info(f"Creating collection: {collection_name}")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                f"text-dense-{size}": models.VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    )
                ) for size in ["128", "256", "512"]
            }
        )
        logger.info("Collection created successfully")

        # ====================================================================
        # SETUP
        # ====================================================================
        text_splitters = create_text_splitters()
        
        # ✅ KEY PARAMETERS
        EMBEDDING_BATCH_SIZE = 50  # Số nodes để embed cùng lúc
        UPLOAD_BATCH_SIZE = 8      # Số points để upload mỗi lần
        MAX_RETRIES = 3
        RETRY_DELAY = 5
        
        # Statistics
        total_uploaded = 0
        total_failed = 0
        context_stats = {
            "total": 0,
            "with_both": 0,
            "with_before_only": 0,
            "with_after_only": 0,
            "without_context": 0
        }
        
        # ====================================================================
        # STREAMING PROCESS: Process nodes in chunks
        # ====================================================================
        logger.info("=" * 80)
        logger.info(f"STREAMING MODE: Processing {len(nodes)} nodes")
        logger.info(f"  Embedding batch size: {EMBEDDING_BATCH_SIZE}")
        logger.info(f"  Upload batch size: {UPLOAD_BATCH_SIZE}")
        logger.info("=" * 80)
        
        total_chunks = (len(nodes) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        
        for chunk_idx in range(0, len(nodes), EMBEDDING_BATCH_SIZE):
            chunk_nodes = nodes[chunk_idx:chunk_idx + EMBEDDING_BATCH_SIZE]
            current_chunk_num = chunk_idx // EMBEDDING_BATCH_SIZE + 1
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing chunk {current_chunk_num}/{total_chunks} "
                       f"(nodes {chunk_idx}-{chunk_idx + len(chunk_nodes)})")
            logger.info(f"{'=' * 80}")
            
            # ================================================================
            # PHASE 1: Embed this chunk of nodes
            # ================================================================
            logger.info(f"Phase 1: Embedding {len(chunk_nodes)} nodes...")
            chunk_points = []
            
            for node_idx, node in enumerate(chunk_nodes):
                global_idx = chunk_idx + node_idx
                
                try:
                    # Progress
                    if node_idx % 10 == 0:
                        logger.info(f"  Embedding: {node_idx}/{len(chunk_nodes)} "
                                  f"(global: {global_idx}/{len(nodes)})")
                    
                    # Get node content and metadata
                    node_content = node.get_content()
                    metadata = node.metadata
                    
                    # ========================================================
                    # Generate multi-vector embeddings
                    # ========================================================
                    vectors = {}
                    
                    for size, splitter in text_splitters.items():
                        try:
                            # Split into sub-chunks
                            sub_chunks = splitter.split_text(node_content)
                            
                            # Limit sub-chunks to save memory
                            max_subchunks = 10  
                            if len(sub_chunks) > max_subchunks:
                                sub_chunks = sub_chunks[:max_subchunks]
                            
                            embeddings = Settings.embed_model.get_text_embedding_batch(
                                sub_chunks,
                                show_progress=False
                            )
                            
                            vectors[f"text-dense-{size}"] = embeddings
                            
                        except Exception as e:
                            logger.error(f"Error embedding size {size}: {e}")
                            # Fallback
                            embedding = Settings.embed_model.get_text_embedding(node_content)
                            vectors[f"text-dense-{size}"] = [embedding]
                    
                    # ========================================================
                    # Validate and add context
                    # ========================================================
                    def validate_context(ctx):
                        if not ctx or ctx == 'N/A':
                            return 'N/A'
                        if not isinstance(ctx, str):
                            return 'N/A'
                        ctx = ctx.strip()
                        if not ctx or ctx.lower() in ['nan', 'nat', 'none', '']:
                            return 'N/A'
                        return ctx
                    
                    context_before = validate_context(metadata.get('context_before_text', 'N/A'))
                    context_after = validate_context(metadata.get('context_after_text', 'N/A'))
                    
                    has_before = context_before != 'N/A'
                    has_after = context_after != 'N/A'
                    
                    # Update statistics
                    context_stats["total"] += 1
                    if has_before and has_after:
                        context_stats["with_both"] += 1
                    elif has_before:
                        context_stats["with_before_only"] += 1
                    elif has_after:
                        context_stats["with_after_only"] += 1
                    else:
                        context_stats["without_context"] += 1
                    
                    # Build payload
                    payload = {
                        "content": node_content,
                        "context_before_text": context_before,
                        "context_after_text": context_after,
                        **{k: v for k, v in metadata.items() 
                           if k not in ['content', 'context_before_text', 'context_after_text']}
                    }
                    
                    # Create point
                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        payload=payload,
                        vector=vectors
                    )
                    
                    chunk_points.append(point)
                    
                except Exception as e:
                    logger.error(f"Error processing node {global_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    total_failed += 1
                    continue
            
            logger.info(f"✓ Embedded {len(chunk_points)} points for this chunk")
            
            # ================================================================
            # PHASE 2: Upload this chunk's points in mini-batches
            # ================================================================
            logger.info(f"Phase 2: Uploading {len(chunk_points)} points...")
            
            num_upload_batches = (len(chunk_points) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
            
            for upload_idx in range(0, len(chunk_points), UPLOAD_BATCH_SIZE):
                upload_batch = chunk_points[upload_idx:upload_idx + UPLOAD_BATCH_SIZE]
                upload_batch_num = upload_idx // UPLOAD_BATCH_SIZE + 1
                
                logger.info(f"  Upload batch {upload_batch_num}/{num_upload_batches} "
                          f"({len(upload_batch)} points)")
                
                # Retry logic
                upload_success = False
                for attempt in range(MAX_RETRIES):
                    try:
                        client.upsert(
                            collection_name=collection_name,
                            points=upload_batch,
                            wait=True, 
                            timeout=60
                        )
                        total_uploaded += len(upload_batch)
                        upload_success = True
                        logger.info(f"    ✓ Upload successful")
                        break
                        
                    except Exception as e:
                        logger.warning(f"    Upload attempt {attempt + 1}/{MAX_RETRIES} failed: {str(e)}")
                        
                        if attempt < MAX_RETRIES - 1:
                            import time
                            logger.info(f"    Retrying in {RETRY_DELAY}s...")
                            time.sleep(RETRY_DELAY)
                        else:
                            logger.error(f"    Failed after {MAX_RETRIES} attempts")
                            total_failed += len(upload_batch)
                
                if not upload_success:
                    logger.error(f"  ✗ Failed to upload batch {upload_batch_num}")
            
            # ================================================================
            # CLEANUP: Clear memory after each chunk
            # ================================================================
            del chunk_points
            del chunk_nodes
            import gc
            gc.collect()
            
            logger.info(f"✓ Chunk {current_chunk_num} completed. "
                       f"Progress: {total_uploaded}/{len(nodes)} uploaded, "
                       f"{total_failed} failed")

        # ====================================================================
        # FINAL STATISTICS
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("INDEXING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total nodes processed: {len(nodes)}")
        logger.info(f"Successfully uploaded: {total_uploaded}")
        logger.info(f"Failed: {total_failed}")
        
        logger.info("\n" + "=" * 80)
        logger.info("CONTEXT STATISTICS")
        logger.info("=" * 80)
        total = context_stats["total"]
        if total > 0:
            logger.info(f"Total nodes: {total}")
            logger.info(f"With both contexts: {context_stats['with_both']} "
                       f"({context_stats['with_both']/total*100:.1f}%)")
            logger.info(f"With before only: {context_stats['with_before_only']} "
                       f"({context_stats['with_before_only']/total*100:.1f}%)")
            logger.info(f"With after only: {context_stats['with_after_only']} "
                       f"({context_stats['with_after_only']/total*100:.1f}%)")
            logger.info(f"Without context: {context_stats['without_context']} "
                       f"({context_stats['without_context']/total*100:.1f}%)")
        logger.info("=" * 80)

        # ====================================================================
        # VERIFICATION
        # ====================================================================
        try:
            sample_points = client.scroll(collection_name, limit=10)[0]
            with_context = sum(
                1 for p in sample_points
                if p.payload.get('context_before_text', 'N/A') != 'N/A'
                or p.payload.get('context_after_text', 'N/A') != 'N/A'
            )
            logger.info(f"\nVerification: {with_context}/10 samples have context")
            
            if sample_points:
                logger.info(f"Sample payload keys: {list(sample_points[0].payload.keys())}")
                if hasattr(sample_points[0], 'vector'):
                    vector_keys = list(sample_points[0].vector.keys()) if isinstance(sample_points[0].vector, dict) else []
                    logger.info(f"Vector keys: {vector_keys}")
        except Exception as e:
            logger.warning(f"Verification failed: {e}")

        logger.info(f"\n✓ Successfully created and populated collection: {collection_name}")
        return collection_name
        
    except Exception as e:
        logger.error(f"Error creating multi-vector index: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_and_populate_bm25_index(nodes: List[Any], persist_path: str) -> None:
    """Create and persist BM25 index."""
    from llama_index.retrievers.bm25 import BM25Retriever
    
    logger.info(f"Creating BM25 index at: {persist_path}")
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )
    
    os.makedirs(persist_path, exist_ok=True)
    bm25_retriever.persist(persist_path)
    
    logger.info(f"✓ BM25 index created and persisted to {persist_path}")


def get_user_confirmation(collection_name: str = None) -> str:
    """Get user confirmation for processing."""
    if collection_name:
        message = f"Do you want to replace the existing collection '{collection_name}'?"
    else:
        message = "Do you want to create the unified index?"
    
    while True:
        response = input(f"{message} (yes/no): ").lower()
        if response in ['yes', 'y', 'no', 'n']:
            return response
        print("Please answer 'yes' or 'no'.")


def main():
    """
    Main indexing function.
    WORKFLOW:
    1. Setup embedding model
    2. Load metadata and documents (pre-split)
    3. Run pipeline (NO splitting)
    4. Create Qdrant index with multi-vector + pre-computed context
    5. Create BM25 index
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING INDEXING WITH PRE-COMPUTED CONTEXT")
        logger.info("=" * 80)
        
        # ====================================================================
        # STEP 1: SETUP EMBEDDING MODEL
        # ====================================================================
        logger.info("\nStep 1: Setting up embedding model...")
        embed_model = setup_embedding_model()
        Settings.embed_model = embed_model
        logger.info(f"✓ Embedding model loaded: {embed_model}")
        
        # ====================================================================
        # STEP 2: LOAD METADATA AND DOCUMENTS
        # ====================================================================
        logger.info("\nStep 2: Loading metadata and documents...")
        metadata_df = load_metadata(DOC_METADATA_PATH)
        documents = load_documents(DOC_DIR_PATH, metadata_df)
        
        if not documents:
            logger.error("No documents loaded. Exiting.")
            return
        
        logger.info(f"✓ Loaded {len(documents)} pre-split documents")
        
        # ====================================================================
        # STEP 3: RUN INGESTION PIPELINE (NO SPLITTING)
        # ====================================================================
        logger.info("\nStep 3: Running ingestion pipeline (no splitting)...")
        pipeline = setup_ingestion_pipeline(INGESTION_CACHE_PATH)
        
        nodes = pipeline.run(documents=documents, show_progress=True)
        
        try:
            pipeline.cache.persist(INGESTION_CACHE_PATH)
            logger.info("✓ Cache persisted")
        except Exception as e:
            logger.warning(f"Could not persist cache: {e}")
        
        logger.info(f"✓ Created {len(nodes)} nodes (1:1 with documents)")
        
        # Filter empty nodes
        logger.info("Filtering out empty or whitespace-only nodes...")
        original_node_count = len(nodes)
        nodes = [node for node in nodes if node.get_content().strip()]
        filtered_count = original_node_count - len(nodes)

        if filtered_count > 0:
            logger.warning(f"Removed {filtered_count} empty nodes.")
        logger.info(f"✓ Remaining nodes for indexing: {len(nodes)}")

        # Statistics
        topics = {}
        for node in nodes:
            topic = node.metadata.get('topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
        logger.info(f"✓ Node distribution by topic: {topics}")

        # ====================================================================
        # STEP 4: CREATE QDRANT INDEX
        # ====================================================================
        logger.info("\nStep 4: Creating Qdrant index with multi-vector + context...")
        
        client = QdrantClient(host="localhost", port=6333, timeout=600)

        try:
            collection_name = QDRANT_COLLECTION_NAME
            multi_vector_index = None
            
            if client.collection_exists(collection_name):
                if get_user_confirmation(collection_name) in ['yes', 'y']:
                    logger.info(f"Replacing existing collection: {collection_name}")
                    client.delete_collection(collection_name)
                    multi_vector_index = create_and_populate_multi_vector_index(nodes, client)
                else:
                    logger.info(f"Skipping replacement of collection: {collection_name}")
            else:
                if get_user_confirmation() in ['yes', 'y']:
                    logger.info(f"Creating new collection: {collection_name}")
                    multi_vector_index = create_and_populate_multi_vector_index(nodes, client)
                else:
                    logger.info("Indexing cancelled by user.")

            # ================================================================
            # STEP 5: CREATE BM25 INDEX
            # ================================================================
            if multi_vector_index is not None:
                logger.info("\nStep 5: Creating BM25 index...")
                bm25_path = os.path.join(BM25_PERSIST_PATH, "unified")
                create_and_populate_bm25_index(nodes, bm25_path)
            else:
                logger.warning("Skipping BM25 index creation.")

            # ================================================================
            # COMPLETION
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("✓ INDEXING COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            if multi_vector_index:
                logger.info(f"\nCollection: {multi_vector_index}")
                logger.info(f"Total nodes: {len(nodes)}")
                logger.info(f"Features:")
                logger.info(f"  - Multi-vector (128, 256, 512)")
                logger.info(f"  - Pre-computed context windows")
                logger.info(f"  - BM25 index")
                
        finally:
            client.close()
            logger.info("Qdrant client closed")
        
    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error("✗ CRITICAL ERROR IN MAIN FUNCTION")
        logger.error(f"{'=' * 80}")
        logger.error(f"Error: {str(e)}")
        
        import traceback
        logger.error("\nFull traceback:")
        logger.error(traceback.format_exc())
        
        raise


if __name__ == "__main__":
    main()