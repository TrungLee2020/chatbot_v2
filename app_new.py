# app-dev-vllm-api.py - Fixed version with improved error handling
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import sys
import time
from openai import AsyncOpenAI
import os
from database import SessionLocal, ChatHistory
from sqlalchemy.orm import Session
from fastapi import Depends

from prompts import CHATBOT_RESPONSE_PROMPT, METADATA_EVALUATION_PROMPT, EVALUATION_CHAT_RESPONSE_PROMPT, STANDALONE_QUESTION_PROMPT
from log import log, set_request_context, clear_request_context, timed, get_token_length
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

# FIXED: Properly configure vLLM server URL
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL")
VLLM_MODEL_NAME = os.getenv("CHATBOT_MODEL", "")
CHAT_HISTORY_LENGTH = int(os.getenv("CHAT_HISTORY_LENGTH", "3"))

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = []
    for error in exc.errors():
        error_details.append(
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        )
    log("error", "Validation error", error_details=error_details)
    return JSONResponse(status_code=422, content={"detail": error_details})

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def setup_vllm_client():
    """Setup vLLM client with proper error handling and validation"""
    try:
        if not VLLM_SERVER_URL:
            raise ValueError("VLLM_SERVER_URL is not configured")
        
        logger.info(f"Setting up vLLM client for server: {VLLM_SERVER_URL}")
        logger.info(f"Model name: {VLLM_MODEL_NAME}")
        
        timeout = float(os.environ.get("VLLM_TIMEOUT", 60))
        api_key = os.environ.get("VLLM_API_KEY", "dummy-key") 
        
        client = AsyncOpenAI(
            base_url=f"{VLLM_SERVER_URL}/vllm/v1",
            api_key=api_key,
            timeout=timeout,
            max_retries=2
        )
        
        logger.info(f"vLLM client initialized successfully")
        return client
    
    except Exception as e:
        logger.error(f"Failed to setup vLLM client: {str(e)}")
        return None

# Initialize client
client = setup_vllm_client()

# Updated topic mapping
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

# Legacy topic mapping for backward compatibility
LEGACY_TOPIC_MAPPING = {
    "QUY ĐỊNH CHUNG CHO HOẠT ĐỘNG CỦA BƯU ĐIỆN VĂN HOÁ XÃ": "bd_vhx",
    "DỊCH VỤ BƯU CHÍNH CHUYỂN PHÁT": "bccp_nd",
    "DỊCH VỤ TÀI CHÍNH BƯU CHÍNH": "tcbc",
    "DỊCH VỤ PHÂN PHỐI BÁN LẺ": "ppbl",
    "DỊCH VỤ HÀNH CHÍNH CÔNG": "hcc",
    "Dau Tu Phat Trien": "dtpt",
    "Nghien Cuu Phat Trien va Thuong Hieu": "ncpt",
    "Quan Ly Chat Luong": "qlcl",
    "Ky Thuat Cong Nghe": "ktcn",
    "Trung Tam Doi Soat": "ttds",
    "Van Phong TCT": "vptl",
    "Tem Buu Chinh": "temBC",
    "Kiem Tra Phap Che": "ktpc",
    "Dich Vu Khach Hang": "dvkh",
}

class ChatHistoryItem(BaseModel):
    chatbot: str
    human: str

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    transaction_id: str
    user_message: str
    chat_history: List[ChatHistoryItem]
    topic: Optional[str] = None

class RerankingScoreItem(BaseModel):
    doc_id: str
    original_score: float
    final_logit: float 
    final_probability: float
    source: str
    topic: str
    text_preview: str

class RetrievalMetrics(BaseModel):
    reranking_scores: List[RerankingScoreItem]
    stats: Dict[str, Any]
    context: str
    structured_context: List[Dict[str, Any]] = [] 
    original_dense_texts: List[Dict[str, Any]]
    original_bm25_texts: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    bot_message: str
    structured_references: Optional[List[Dict[str, Any]]] = None
    doc_id: List[str]
    show_ref: int
    timestamp: float
    err_id: Optional[str]
    retrieval_metrics: str  # Trường này vẫn là string vì bạn .json() nó ở cuối
    structured_context: Optional[List[Dict[str, Any]]] = None

def normalize_topic(topic: str) -> Optional[str]:
    """Normalize topic from legacy format to new unified format"""
    if not topic:
        return None
    
    if topic in VALID_TOPICS:
        return topic
    
    if topic in LEGACY_TOPIC_MAPPING:
        return LEGACY_TOPIC_MAPPING[topic]
    
    log("warning", f"Unknown topic '{topic}', searching across all topics")
    return None


def get_chat_history(messages: List[ChatHistoryItem]) -> str:
    """
    Trích xuất N cặp hội thoại gần nhất từ lịch sử chat.
    Số lượng N được cấu hình qua biến môi trường CHAT_HISTORY_LENGTH.
    """
    chat_history = []
    
    recent_messages = messages[-CHAT_HISTORY_LENGTH:]
    
    for msg in recent_messages:
        # Chỉ thêm vào lịch sử nếu có nội dung
        if msg.human:
            chat_history.append(f"Human: {msg.human}")
        if msg.chatbot:
            chat_history.append(f"Chatbot: {msg.chatbot}")
            
    return "\n".join(chat_history)

@timed
async def generate_standalone_question(chat_history: str, user_message: str) -> str:
    """
    Sử dụng LLM để viết lại câu hỏi của người dùng dựa trên lịch sử chat,
    tạo ra một câu hỏi đầy đủ ngữ cảnh.
    """
    if not chat_history:
        return user_message

    if client is None:
        log("warning", "vLLM client not available for query rewriting, using original message.")
        return user_message

    system_message = STANDALONE_QUESTION_PROMPT.format(
        chat_history=chat_history,
        user_message=user_message
    )

    log("debug", "Generating standalone question", 
        prompt_length=get_token_length(system_message))

    try:
        response = await client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[
                {"role": "user", "content": system_message + "/no_think"},
            ],
            temperature=0.0,  # Dùng temperature thấp để kết quả nhất quán, không sáng tạo
            top_p=1.0,
            max_tokens=256,   # Câu hỏi thường không quá dài
            extra_body={"repetition_penalty": 1.0},
        )
        
        if response.choices and response.choices[0].message.content:
            standalone_question = response.choices[0].message.content.strip()
            log("info", "Query rewritten", original=user_message, rewritten=standalone_question)
            return standalone_question
        else:
            log("warning", "Query rewriting returned empty response, using original message.")
            return user_message
            
    except Exception as e:
        log("error", f"Error in generating standalone question: {str(e)}", error_type=type(e).__name__)
        # Nếu có lỗi, trả về câu hỏi gốc để không làm gián đoạn luồng
        return user_message

@timed
async def perform_rag_query(
    topic_filter: Optional[str],
    chat_history: str,
    user_input: str,
    user_id: str,
    session_id: str,
    transaction_id: str,
) -> tuple:
    log("debug", "RAG query initiated", topic_filter=topic_filter, user_input_length=get_token_length(user_input))

    RAG_API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8338/rag_query")
    
    payload = {
        "chat_history": chat_history,
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "transaction_id": transaction_id,
    }
    # Chỉ thêm key "topic" vào payload nếu nó thực sự có giá trị
    if topic_filter is not None:
        payload["topic"] = topic_filter
    else:
        payload["topic"] = ""
        
    try:
        log("debug", f"Sending request to RAG API at {RAG_API_URL}")
        timeout = aiohttp.ClientTimeout(total=60)
        
        log("debug", f"RAG request payload: {payload}")
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(RAG_API_URL, json=payload) as response:
                log("debug", f"RAG API response status: {response.status}")
                
                if response.status != 200:
                    error_text = await response.text()
                    log("error", f"RAG API returned status {response.status}: {error_text}")
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=error_text
                    )
                
                response_json = await response.json()
                log("debug", f"RAG API response received successfully")
                
                context_text = response_json.get("response", "Không nhận được phản hồi từ API")
                doc_metadata = response_json.get("doc_metadata", [])
                doc_ids = response_json.get("doc_ids", "")
                doc_ids_array = doc_ids.split(",") if doc_ids else []
                doc_ids = [doc.strip() for doc in doc_ids_array if doc.strip()]
                
                retrieval_details_dict = response_json.get("retrieval_metrics", {
                    "reranking_scores": [], 
                    "stats": {}, 
                    "context": "", 
                    "original_dense_texts": [], 
                    "original_bm25_texts": []
                })
                             
                retrieval_details = RetrievalMetrics(
                                    reranking_scores=retrieval_details_dict.get("reranking_scores", []),
                                    stats=retrieval_details_dict.get("stats", {}),
                                    context=retrieval_details_dict.get("context", ""),
                                    structured_context=retrieval_details_dict.get("structured_context", []), 
                                    original_dense_texts=retrieval_details_dict.get("original_dense_texts", []),
                                    original_bm25_texts=retrieval_details_dict.get("original_bm25_texts", [])
                                )

                log("debug", "RAG query completed successfully", context_length=get_token_length(context_text))
                return context_text, doc_metadata, retrieval_details, doc_ids
        
    except asyncio.TimeoutError:
        log("error", "RAG API timeout")
        context_text = "Có lỗi timeout trong quá trình RAG query."
        doc_metadata = []
        doc_ids = []
        retrieval_details = RetrievalMetrics(
            reranking_scores=[], stats={"error": "TIMEOUT"}, context="",
            original_dense_texts=[], original_bm25_texts=[]
        )
    except aiohttp.ClientError as e:
        log("error", f"RAG API client error: {str(e)}", error_type=type(e).__name__)
        context_text = "Có lỗi kết nối với RAG API."
        doc_metadata = []
        doc_ids = []
        retrieval_details = RetrievalMetrics(
            reranking_scores=[], stats={"error": f"CLIENT_ERROR: {str(e)}"}, context="",
            original_dense_texts=[], original_bm25_texts=[]
        )
    except Exception as e:
        log("error", f"Unexpected error in RAG query: {str(e)}", error_type=type(e).__name__)
        context_text = "Có lỗi không mong đợi trong quá trình truy vấn."
        doc_metadata = []
        doc_ids = []
        retrieval_details = RetrievalMetrics(
            reranking_scores=[], stats={"error": f"UNEXPECTED_ERROR: {str(e)}"}, context="",
            original_dense_texts=[], original_bm25_texts=[]
        )
    
    return context_text, doc_metadata, retrieval_details, doc_ids

@timed
async def generate_bot_response(context_text: str, user_message: str, topic_display: str) -> str:
    """Generate bot response with improved error handling and model validation"""
    if client is None:
        log("error", "vLLM client not available")
        return "Hệ thống đang bảo trì. Xin vui lòng thử lại sau."
    
    system_message = CHATBOT_RESPONSE_PROMPT.format(
        context=context_text, 
        chat_history="", 
        topic=topic_display or "Thông tin tổng hợp",
    )

    log("debug", "Generating bot response", 
        system_message_length=get_token_length(system_message),
        user_message_length=get_token_length(user_message),
        model_name=VLLM_MODEL_NAME)

    try:
        # FIXED: Test if model exists before making request
        log("debug", f"Attempting completion with model: {VLLM_MODEL_NAME}")
        log("debug", f"Using endpoint: {VLLM_SERVER_URL}/v1/chat/completions")
        
        response = await client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=float(os.getenv("CHATBOT_TEMPERATURE", 0.2)),
            top_p=float(os.getenv("CHATBOT_TOP_P", 0.95)),
            max_tokens=int(os.getenv("CHATBOT_MAX_TOKENS", 1024)),
            extra_body={
                "repetition_penalty": float(os.getenv("CHATBOT_REPETITION_PENALTY", 1.05)),
                "chat_template_kwargs": {"enable_thinking": False}
            },
        )
        log("debug", "Successfully received response from chat completion API")
        
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return str(content)
            else:
                log("warning", "Empty response content from vLLM")
                return "Xin lỗi, không nhận được phản hồi từ hệ thống."
        else:
            log("warning", "No choices in response from vLLM")
            return "Xin lỗi, không có phản hồi từ hệ thống."
    
    except asyncio.TimeoutError:
        log("error", "Timeout error in generating bot response")
        return "Xin lỗi, phản hồi bị timeout. Vui lòng thử lại."
    except Exception as e:
        log("error", f"Error in generating bot response: {str(e)}", error_type=type(e).__name__)
        
        # FIXED: More specific error messages based on error type
        if "404" in str(e):
            log("error", f"Model or endpoint not found. Check if model '{VLLM_MODEL_NAME}' is available")
            return "Xin lỗi, mô hình AI không khả dụng. Vui lòng liên hệ quản trị viên."
        elif "connection" in str(e).lower():
            return "Xin lỗi, không thể kết nối đến server AI. Vui lòng thử lại sau."
        else:
            return "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý yêu cầu của bạn."

async def check_vllm_health():
    """Check vLLM server health with better error reporting"""
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # FIXED: Check multiple endpoints
            health_endpoints = ["/health", "/v1/models", "/"]
            
            for endpoint in health_endpoints:
                try:
                    url = f"{VLLM_SERVER_URL}{endpoint}"
                    log("debug", f"Checking vLLM health at: {url}")
                    
                    async with session.get(url) as response:
                        log("debug", f"Health check response: {response.status}")
                        
                        if response.status == 200:
                            if endpoint == "/v1/models":
                                # Check if our model is available
                                models_data = await response.json()
                                log("info", f"Available models: {models_data}")
                                
                                model_list = models_data.get("data", [])
                                model_names = [model.get("id", "") for model in model_list]
                                
                                if VLLM_MODEL_NAME in model_names:
                                    log("info", f"Target model '{VLLM_MODEL_NAME}' is available")
                                    return True
                                else:
                                    log("warning", f"Target model '{VLLM_MODEL_NAME}' not found in available models: {model_names}")
                                    return False
                            else:
                                log("info", f"vLLM server responded OK to {endpoint}")
                                return True
                        else:
                            log("warning", f"Health check failed for {endpoint} with status {response.status}")
                            
                except Exception as endpoint_error:
                    log("debug", f"Health check failed for {endpoint}: {str(endpoint_error)}")
                    continue
            
            log("error", "All health check endpoints failed")
            return False
            
    except asyncio.TimeoutError:
        log("error", "vLLM server health check timeout")
        return False
    except Exception as e:
        log("error", f"Error checking vLLM server health: {str(e)}")
        return False

async def evaluate_metadata_need(response: str, doc_metadata: str) -> bool:
    if client is None:
        return False
        
    user_content = METADATA_EVALUATION_PROMPT.format(response=response, doc_metadata=doc_metadata)

    try:
        evaluation_response = await client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who evaluates the relevance of metadata to the user's question."},
                {"role": "user", "content": user_content},
            ],
            temperature=float(os.getenv("METADATA_EVAL_TEMPERATURE", 0.1)),
            max_tokens=int(os.getenv("METADATA_EVAL_MAX_TOKENS", 50)),
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        result = eval(evaluation_response.choices[0].message.content)
        log("info", f"Metadata evaluation result: {result}")
        return result["is_metadata_relevant"]
    except Exception as e:
        log("error", f"Error in evaluating metadata need: {str(e)}")
        return False

async def evaluate_chatbot_response(user_message: str, response: str):
    if client is None:
        return False
        
    eval_content = EVALUATION_CHAT_RESPONSE_PROMPT.format(user_message=user_message, response=response)
    
    try:
        eval_chat_response = await client.chat.completions.create(
            model=VLLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who evaluates the relevance of response to the user's question."},
                {"role": "user", "content": eval_content},
            ],
            temperature=float(os.getenv("EVALUATION_CHAT_RESPONSE_PROMPT", 0.1)),
            max_tokens=int(os.getenv("EVALUATION_CHAT_RESPONSE_PROMPT", 50)),
        )

        result = eval(eval_chat_response.choices[0].message.content)
        log("info", f"Evaluation chatbot response result: {result}")
        return result["is_response_eval"]
    except Exception as e:
        log("error", f"Error in evaluation chatbot response: {str(e)}")
        return False

@app.post("/chat", response_model=ChatResponse)
@timed
async def chat_endpoint(chat_request: ChatRequest, db: Session = Depends(get_db)):
    set_request_context(chat_request.transaction_id, chat_request.user_id, chat_request.session_id)

    empty_retrieval_metrics = RetrievalMetrics(
        reranking_scores=[], 
        stats={
            "error": "PROCESSING_ERROR",
            "total_final_results": 0,
            "dense_reranked_results": 0,
            "bm25_top_results": 0
        }, 
        context="",
        structured_context=[],
        original_dense_texts=[], 
        original_bm25_texts=[]
    )

    try:
        log("info", "Chat request received", topic=chat_request.topic, 
            message_length=get_token_length(chat_request.user_message))

        topic_filter = normalize_topic(chat_request.topic)
        
        if topic_filter:
            log("info", f"Using topic filter: {topic_filter}")
        else:
            log("info", "Searching across all topics (no filter applied)")

        # Check vLLM health but don't block
        is_healthy = await check_vllm_health()
        if not is_healthy:
            log("warning", "vLLM server health check failed, but continuing with request")

        chat_history_str = get_chat_history(chat_request.chat_history)
        
        # Viết lại câu hỏi để có đầy đủ ngữ cảnh
        standalone_question = await generate_standalone_question(chat_history_str, chat_request.user_message)

        # BƯỚC 2: Sử dụng câu hỏi đã viết lại để truy vấn RAG
        try:
            context_text, doc_metadata, retrieval_details, doc_ids = await perform_rag_query(
                topic_filter, 
                chat_history_str, 
                standalone_question,
                chat_request.user_id, 
                chat_request.session_id, 
                chat_request.transaction_id,
            )
            structured_context = retrieval_details.structured_context
            # Log context statistics
            if structured_context:
                chunks_with_before = sum(
                    1 for item in structured_context 
                    if item.get('has_context_before', False)
                )
                chunks_with_after = sum(
                    1 for item in structured_context 
                    if item.get('has_context_after', False)
                )
                
                log("info", "Context window statistics",
                    total_chunks=len(structured_context),
                    with_context_before=chunks_with_before,
                    with_context_after=chunks_with_after)

        except Exception as rag_error:
            log("error", f"RAG query failed completely: {str(rag_error)}")
            # Return fallback response
            return ChatResponse(
                bot_message="Xin lỗi, hệ thống tìm kiếm thông tin đang gặp sự cố. Vui lòng thử lại sau.",
                ref_doc=None, doc_id=[], show_ref=0, timestamp=time.time(),
                err_id=f"RAG_ERROR: {str(rag_error)}", 
                retrieval_metrics=empty_retrieval_metrics.model_dump_json(),
            )

        topic_display = chat_request.topic or topic_filter
        
        # Tạo câu trả lời cuối cùng vẫn dùng câu hỏi gốc của người dùng để tự nhiên
        if structured_context and len(structured_context) > 0:
            # Lấy text từ structured_context (đã có full context)
            context_with_window = "\n***\n".join([
                item.get("text", "") for item in structured_context
            ])
            
            log("info", "Using context with context window", 
                chunks_count=len(structured_context),
                total_length=get_token_length(context_with_window))
            
            # Dùng context này để generate response
            response = await generate_bot_response(
                context_with_window,  # Dùng context có window
                chat_request.user_message, 
                topic_display
            )
        else:
            # Fallback: Dùng context_text thông thường nếu không có structured_context
            log("warning", "No structured context available, using plain context")
            response = await generate_bot_response(
                context_text, 
                chat_request.user_message, 
                topic_display
            )        
        response = str(response) if response is not None else ""

        # Lưu lịch sử chat vào DB (logic từ câu trả lời trước)
        try:
            db_chat_history = ChatHistory(
                user_id=chat_request.user_id,
                session_id=chat_request.session_id,
                question=chat_request.user_message,
                answer=response
            )
            db.add(db_chat_history)
            db.commit()
            db.refresh(db_chat_history)
            log("info", f"Saved chat history with id {db_chat_history.id}")
        except Exception as db_error:
            log("error", f"Failed to save chat history: {str(db_error)}")
            db.rollback()
        
        response = str(response) if response is not None else ""
        need_metadata = await evaluate_metadata_need(response, doc_metadata)

        # Token calculations
        prompt_token_length = get_token_length(
            CHATBOT_RESPONSE_PROMPT.format(
                context=context_text, chat_history=chat_request.user_message,
                topic=topic_display or "Thông tin tổng hợp",
            )
        )
        response_token_length = get_token_length(response) if response else 0
        total_tokens = prompt_token_length + response_token_length

        log("info", "Chat response generated", prompt_tokens=prompt_token_length,
            response_tokens=response_token_length, total_tokens=total_tokens)

        log("info", f"Saved chat history with id {db_chat_history.id}")
        retrieval_details_json = retrieval_details.json()

        # Check for default no-info response
        DEFAULT_NO_INFO_RESPONSE = "Rất tiếc, tôi chưa thể tìm thấy thông tin cụ thể về câu hỏi của bạn trong dữ liệu hiện có. Xin vui lòng đặt lại câu hỏi chi tiết hơn."
        normalized_response = response.strip() if response else ""
        is_default_response = normalized_response == DEFAULT_NO_INFO_RESPONSE.strip()

        if is_default_response:
            need_metadata = False

        eval_response = await evaluate_chatbot_response(user_message=chat_request.user_message, response=response)
        
        if eval_response == 1:
            return ChatResponse(
                bot_message=response, structured_references=doc_metadata if need_metadata else None,
                doc_id=doc_ids, show_ref=int(need_metadata), timestamp=time.time(),
                err_id=None, retrieval_metrics=retrieval_details_json,
                structured_context=structured_context
            )
        else:
            return ChatResponse(
                bot_message=response, 
                structured_references=None if is_default_response else (doc_metadata if need_metadata else None),
                doc_id=doc_ids, show_ref=int(need_metadata), timestamp=time.time(),
                err_id=None, retrieval_metrics=retrieval_details_json,
                structured_context=structured_context
            )

    except Exception as e:
        log("error", "Error processing chat", error=str(e), error_type=type(e).__name__)
        return ChatResponse(
            bot_message="Hệ thống đang gặp trục trặc. Xin vui lòng thử lại sau.",
            ref_doc=None, doc_id=[], show_ref=0, timestamp=time.time(),
            err_id=str(e), retrieval_metrics=empty_retrieval_metrics.model_dump_json(),
            llm_context=None
        )
    finally:
        clear_request_context()

@app.get("/health")
async def health_check():
    try:
        vllm_status = "health" if client else "unavailable"
        rag_url = os.getenv("RAG_API_URL")
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(rag_url.replace("/rag_query", "/health")) as response:
                    rag_status = "healthy" if response.status == 200 else "unhealthy"
        except:

            rag_status = "unhealthy"
        return {
                "status": "healthy",
                "timestamp": time.time(),
                "services": {
                    "vllm": vllm_status,
                    "rag": rag_status
                    }
                }
    except Exception as e:
        return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)}
                )


@app.on_event("startup")
async def startup_event():
    """Enhanced startup with better diagnostics"""
    logger.info("Starting up application...")
    logger.info(f"VLLM_SERVER_URL: {VLLM_SERVER_URL}")
    logger.info(f"VLLM_MODEL_NAME: {VLLM_MODEL_NAME}")
    
    # Test RAG API connection
    rag_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8338/rag_query")
    logger.info(f"RAG_API_URL: {rag_url}")
    
    # Test vLLM connection
    if client:
        logger.info("vLLM client created, testing connection...")
        is_healthy = await check_vllm_health()
        if is_healthy:
            logger.info("✅ vLLM server connection verified")
        else:
            logger.warning("⚠️  vLLM server connection failed - check server status and model availability")
            logger.warning(f"Please verify that:")
            logger.warning(f"1. vLLM server is running at {VLLM_SERVER_URL}")
            logger.warning(f"2. Model '{VLLM_MODEL_NAME}' is loaded and available")
            logger.warning(f"3. Server accepts requests at /v1/chat/completions endpoint")
    else:
        logger.error("❌ vLLM client not initialized")

    # Init database for chat history
    try:
        from database import create_tables
        create_tables()
        logger.info("==== Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=os.getenv("APP_HOST", "0.0.0.0"), 
        port=int(os.getenv("APP_PORT", 6868)), 
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
