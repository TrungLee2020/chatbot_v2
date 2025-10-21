# streamlit_app.py - Updated for unified collection architecture with new logic

import streamlit as st
import requests
import uuid
import time
import json
import re
from datetime import datetime

# --- CẤU HÌNH ---
API_BASE_URL = "http://localhost:6868"  # Updated to match your app.py port

# Topic options for the unified system
TOPIC_OPTIONS = {
    "Tất cả chủ đề": None,  # No filter - search across all topics
    "Tài chính nhân sự": "tcns",
    "Đầu tư phát triển": "dtpt",
    "Quản lý chất lượng": "qlcl",
    "Kỹ thuật công nghệ": "ktcn",
    "Nghiên cứu phát triển": "ncpt",
    "Kiểm tra pháp chế": "ktpc",
    "Văn phòng TCT": "vptl",
    "Tem bưu chính": "temBC",
    "Trung tâm đối soát": "ttds",
    "Dịch vụ khách hàng": "dvkh",
    "Bưu điện văn hóa xã": "bd_vhx",
    "Bưu chính chuyển phát (nội địa)": "bccp_nd",
    "Bưu chính chuyển phát (quốc tế)": "bccp_qt",
    "Dịch vụ hành chính công": "hcc",
    "Dịch vụ phân phối bán lẻ": "ppbl",
    "Tài chính bưu chính": "tcbc",
}

# --- KHỞI TẠO SESSION STATE ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []

if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs = []

if 'use_uploaded_files' not in st.session_state:
    st.session_state.use_uploaded_files = False

if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = "Tất cả chủ đề"

if 'show_retrieval_details' not in st.session_state:
    st.session_state.show_retrieval_details = False

# --- HELPER FUNCTIONS ---
def format_date_string(date_str):
    """Parse various date string formats and return 'dd-mm-YYYY'."""
    if not date_str or not isinstance(date_str, str) or "N/A" in date_str:
        return None
    try:
        # Tách phần ngày ra khỏi phần thời gian nếu có (ví dụ: '2021-06-01 00:00:00')
        date_part = date_str.split(' ')[0]
        # Các định dạng có thể gặp
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                dt_obj = datetime.strptime(date_part, fmt)
                return dt_obj.strftime('%d-%m-%Y')
            except ValueError:
                continue
    except Exception:
        return None # Trả về None nếu có lỗi
    return None

def refresh_documents():
    """Cập nhật danh sách tài liệu từ server"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{st.session_state.session_id}")
        if response.status_code == 200:
            st.session_state.uploaded_docs = response.json().get("documents", [])
        else:
            st.error(f"Lỗi lấy danh sách tài liệu: {response.text}")
    except Exception as e:
        st.error(f"Lỗi kết nối: {e}")

def get_document_status(doc_id):
    """Lấy trạng thái chi tiết của một tài liệu"""
    try:
        response = requests.get(f"{API_BASE_URL}/document-status/{st.session_state.session_id}/{doc_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def send_chat_request(user_message, topic_filter=None):
    """Gửi yêu cầu chat đến API với topic filter"""
    # Chuẩn bị chat history cho API
    chat_history_for_api = []
    for msg in st.session_state.chat_history[-8:]:  # Lấy 4 cặp gần nhất
        if msg["role"] == "user":
            chat_history_for_api.append({
                "human": msg["content"],
                "chatbot": ""
            })
        elif msg["role"] == "assistant" and chat_history_for_api:
            chat_history_for_api[-1]["chatbot"] = msg["content"]

    # Remove incomplete pairs
    chat_history_for_api = [item for item in chat_history_for_api if item["chatbot"]]

    payload = {
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
        "transaction_id": str(uuid.uuid4()),
        "user_message": user_message,
        "chat_history": chat_history_for_api,
        "topic": topic_filter if topic_filter is not None else ""
    }

    try:
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=120)
        return response
    except Exception as e:
        return None

# --- GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="VNPOST Chatbot", layout="wide")
st.title("🤖 VNPOST ASSISTANT - Unified Search")
st.caption(f"Session ID: {st.session_state.session_id}")

# Refresh danh sách tài liệu khi load page
if st.session_state.uploaded_docs:
    refresh_documents()

# --- SIDEBAR: QUẢN LÝ TÀI LIỆU VÀ CẤU HÌNH ---
with st.sidebar:
    st.header("📂 Quản lý Tài liệu")

    # Upload file section (giữ nguyên logic cũ nếu có)
    st.subheader("🆕 Tải lên tài liệu mới")
    uploaded_file = st.file_uploader(
        "Chọn tài liệu (PDF, DOCX, MD, TXT)",
        type=['pdf', 'docx', 'md', 'txt'],
        key="file_uploader"
    )

    if uploaded_file is not None:
        if st.button("🚀 Xử lý tài liệu", key="process_file"):
            with st.spinner(f"Đang xử lý `{uploaded_file.name}`..."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{API_BASE_URL}/upload",
                        data={'session_id': st.session_state.session_id},
                        files=files,
                        timeout=300
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Đã xử lý: {result['filename']}")
                        refresh_documents()
                        st.rerun()
                    else:
                        st.error(f"❌ Lỗi: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"❌ Lỗi upload: {e}")

    st.divider()

    # Topic Selection for filtering
    st.subheader("🎯 Chọn phạm vi tìm kiếm")
    selected_topic = st.selectbox(
        "Lọc theo chủ đề:",
        options=list(TOPIC_OPTIONS.keys()),
        index=0,  # Default to "Tất cả chủ đề"
        key="topic_selector"
    )
    st.session_state.selected_topic = selected_topic

    topic_filter = TOPIC_OPTIONS[selected_topic]
    if topic_filter:
        st.info(f"🔍 Sẽ tìm kiếm trong: {selected_topic}")
    else:
        st.info("🌐 Tìm kiếm trên tất cả chủ đề")

    st.divider()

    # Advanced options
    st.subheader("⚙️ Cài đặt nâng cao")

    show_details = st.checkbox(
        "Hiển thị chi tiết retrieval",
        value=st.session_state.show_retrieval_details,
        help="Hiển thị thông tin về dense/BM25 retrieval và reranking scores"
    )
    st.session_state.show_retrieval_details = show_details

    # Data source configuration (if you still have document upload feature)
    if st.session_state.uploaded_docs:
        st.subheader("📄 Tài liệu đã upload")
        data_source = st.radio(
            "Nguồn dữ liệu:",
            options=["RAG hệ thống", "Tài liệu upload", "Kết hợp"],
            key="data_source_radio"
        )

        if data_source == "RAG hệ thống":
            st.session_state.use_uploaded_files = False
        elif data_source == "Tài liệu upload":
            st.session_state.use_uploaded_files = True
        else:
            st.session_state.use_uploaded_files = True

        # Document selection (if using uploaded files)
        if st.session_state.use_uploaded_files:
            st.write("Chọn tài liệu:")
            for doc in st.session_state.uploaded_docs:
                doc_id = doc['doc_id']
                filename = doc['filename']
                # <<< THAY ĐỔI 1: Lấy doc_date và hiển thị cùng filename >>>
                doc_date = doc.get('doc_date', 'N/A')
                display_label = f"{filename[:30]}... ({doc_date})"
                is_selected = doc_id in st.session_state.selected_docs

                if st.checkbox(display_label, value=is_selected, key=f"doc_select_{doc_id}"):
                    if doc_id not in st.session_state.selected_docs:
                        st.session_state.selected_docs.append(doc_id)
                else:
                    if doc_id in st.session_state.selected_docs:
                        st.session_state.selected_docs.remove(doc_id)

# --- MAIN CONTENT AREA ---
col1, col2 = st.columns([2, 1])

with col1:
    # Header with clear button
    header_cols = st.columns([0.8, 0.2])
    with header_cols[0]:
        st.header("💬 Khung Chat")
    with header_cols[1]:
        if st.button("🗑️ Xóa", help="Xóa toàn bộ lịch sử chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                structured_references = message.get("structured_references", [])
                structured_context = message.get("structured_context", [])

                if structured_references:
                    st.markdown("**Thông tin tham khảo**")
                    # Dùng st.container để nhóm các expander lại
                    with st.container(border=True):
                        for i, ref in enumerate(structured_references):
                            # Lấy thông tin từ đối tượng tham chiếu (ref)
                            doc_id = ref.get("doc_id")
                            doc_title = ref.get("doc_title")
                            formatted_date = format_date_string(ref.get("doc_date"))

                            # Xây dựng chuỗi hiển thị theo từng phần
                            parts = []
                            if doc_id and doc_id not in ["N/A", "website"]:
                                parts.append(f"Văn bản số {doc_id}")
                            if formatted_date:
                                parts.append(f"ngày {formatted_date}")
                            if doc_title and doc_title not in ["N/A", "website"]:
                                parts.append(f"v/v {doc_title}")

                            # Ghép các phần lại với nhau
                            display_text = " - ".join(parts)

                            # Xử lý trường hợp không có thông tin gì hoặc là N/A
                            if not display_text or doc_id == "N/A":
                                display_text = "N/A (Không có thông tin)"

                            expander_title = f"{i+1}. {display_text}"

                            # SỬA LỖI: Sử dụng biến "doc_id" đã được định nghĩa
                            relevant_chunks = [
                                chunk for chunk in structured_context if chunk.get("doc_id") == doc_id
                            ]
                            
                            with st.expander(expander_title):
                                if relevant_chunks:
                                    st.caption(f"Tìm thấy {len(relevant_chunks)} đoạn context đã được sử dụng từ tài liệu này:")
                                    for j, chunk in enumerate(relevant_chunks):
                                        final_prob = chunk.get('final_probability', 0.0)
                                        st.markdown(f"--- \n *Đoạn {j+1} (Độ liên quan: **{final_prob:.2%}**)*")
                                        st.text_area(
                                            label=f"Chunk Content {j+1}",
                                            value=chunk.get('text', 'N/A'),
                                            height=150,
                                            disabled=True,
                                            # SỬA LỖI: Sử dụng "doc_id" để tạo key duy nhất
                                            key=f"chunk_{message.get('timestamp')}_{doc_id}_{j}"
                                        )
                                else:
                                    st.info("Không có đoạn context cụ thể nào từ tài liệu này được sử dụng trong ngữ cảnh cung cấp cho LLM.")

                # --- KHỐI CODE MỚI: Hiển thị chi tiết retrieval nếu được bật ---
                if st.session_state.show_retrieval_details:
                    details = message.get("retrieval_details", {})
                    reranking_scores = details.get("reranking_scores", [])

                    if reranking_scores:
                        with st.expander("🔍 Chi tiết Retrieval & Reranking"):
                            st.write("Các tài liệu được tìm thấy và điểm số cuối cùng (top kết quả):")

                            # Chuẩn bị dữ liệu để hiển thị dạng bảng
                            display_data = []
                            for item in reranking_scores:
                                # <<< THAY ĐỔI 3: Lấy doc_date và hiển thị trong cột Doc ID >>>
                                doc_id = item.get('doc_id', 'N/A')[:8] # Rút gọn ID
                                doc_date = item.get('doc_date', 'N/A')
                                display_id_with_date = f"{doc_id} ({doc_date})"
                                
                                display_data.append({
                                    "Doc ID": display_id_with_date,
                                    # "Logit": f"{item.get('final_logit', 0.0):.3f}",
                                    "Độ liên quan": f"{item.get('final_probability', 0.0):.2%}",
                                    "Source": item.get('source', 'N/A'),
                                    "Preview": item.get('text_preview', '')[:150] + "..." # Rút gọn preview
                                })

                            # Hiển thị bằng st.dataframe cho đẹp
                            st.dataframe(display_data, use_container_width=True)

                            # Hiển thị thêm các thống kê khác
                            stats = details.get("stats", {})
                            if stats:
                                st.write("**Thống kê Retrieval:**")
                                s_col1, s_col2, s_col3 = st.columns(3)
                                s_col1.metric("Tổng kết quả cuối", stats.get('total_final_results', 0))
                                s_col2.metric("Dense Reranked", stats.get('dense_reranked_results', 0))
                                s_col3.metric("BM25 Top", stats.get('bm25_top_results', 0))

    # Handle assistant response generation
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang suy nghĩ..."):
                try:
                    last_prompt = st.session_state.chat_history[-1]["content"]
                    response = send_chat_request(last_prompt, topic_filter)

                    if response and response.status_code == 200:
                        result = response.json()
                        bot_message = result.get("bot_message", "Xin lỗi, đã có lỗi xảy ra.")
                        structured_references = result.get("structured_references", [])
                        doc_ids = result.get("doc_id", [])
                        retrieval_metrics = result.get("retrieval_metrics", "{}")
                        structured_context = result.get("structured_context", [])

                        try:
                            retrieval_details = json.loads(retrieval_metrics) if isinstance(retrieval_metrics, str) else retrieval_metrics
                        except:
                            retrieval_details = {}

                        # Create assistant message
                        assistant_message = {
                            "role": "assistant",
                            "content": bot_message,
                            "timestamp": str(time.time()),
                            "structured_references": structured_references,
                            "structured_context": structured_context,
                            "retrieval_details": retrieval_details
                        }

                    else:
                        error_msg = f"❌ Lỗi từ API: {response.status_code if response else 'Connection failed'}"
                        if response:
                            error_msg += f" - {response.text}"
                        assistant_message = {
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": str(time.time())
                        }

                except Exception as e:
                    error_msg = f"❌ Lỗi kết nối: {str(e)}"
                    assistant_message = {
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": str(time.time())
                    }

                st.session_state.chat_history.append(assistant_message)
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
            "timestamp": str(time.time())
        })
        st.rerun()

with col2:
    st.header("📊 Thông tin hệ thống")

    # Current configuration
    st.subheader("⚙️ Cấu hình hiện tại")
    st.write(f"**Phạm vi tìm kiếm:** {st.session_state.selected_topic}")

    if topic_filter:
        st.success(f"🎯 Lọc theo: {topic_filter}")
    else:
        st.info("🌐 Tìm kiếm toàn bộ")

    st.write(f"**Chi tiết retrieval:** {'Bật' if st.session_state.show_retrieval_details else 'Tắt'}")

    st.divider()

    # Statistics
    st.subheader("📊 Thống kê")
    st.metric("Tin nhắn trong chat", len(st.session_state.chat_history))
    if st.session_state.uploaded_docs:
        st.metric("Tài liệu đã upload", len(st.session_state.uploaded_docs))
        st.metric("Tài liệu được chọn", len(st.session_state.selected_docs))

    st.divider()

    # Available topics
    st.subheader("📋 Chủ đề có sẵn")
    with st.expander("Xem tất cả chủ đề"):
        for topic_name, topic_code in TOPIC_OPTIONS.items():
            if topic_code:
                st.write(f"• **{topic_name}** (`{topic_code}`)")
            else:
                st.write(f"• **{topic_name}** (tất cả)")

    # System info
    st.subheader("🔧 Thông tin hệ thống")
    st.write("**API Endpoint:** " + API_BASE_URL)
    st.write("**Features:**")
    st.write("• Cross-topic search")
    st.write("• Multi-vector retrieval")
    st.write("• BM25 + Dense fusion (Partial Rerank)") # Ghi chú lại logic mới
    st.write("• Reranking")

# --- FOOTER ---
st.divider()
st.caption("🚀 VNPOST AI Assistant")