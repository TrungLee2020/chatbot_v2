# Chatbot V2 - Refactoring & Improvements Guide

> **Status**: ✅ Refactored architecture complete, ready for RAG improvements implementation
>
> **Last updated**: 2025-10-21

---

## 📚 DOCUMENTS QUAN TRỌNG

### 1. **REFACTOR_NOTES.md** - Architecture Documentation
📍 **Đọc đầu tiên để hiểu cấu trúc mới**

**Nội dung:**
- Giải thích architecture mới (layered design)
- So sánh TRƯỚC vs SAU refactoring
- Hướng dẫn navigation trong codebase
- Migration status
- Cách chạy ứng dụng

**Khi nào đọc:** Trước khi bắt đầu code bất kỳ thứ gì

---

### 2. **RAG_IMPROVEMENTS_PROPOSAL.md** - Implementation Guide
📍 **Hướng dẫn chi tiết 8 improvements cho RAG**

**Nội dung:**
- **Phần 1**: Chunking Improvements (2 proposals)
  - Adaptive chunking theo topic
  - Semantic chunking với sentence boundaries

- **Phần 2**: Vector DB Organization (2 proposals)
  - Hybrid sparse + dense embeddings
  - Topic-partitioned collections (optional)

- **Phần 3**: Retrieval Improvements (2 proposals)
  - Smart retrieval với query classification
  - Topic-adaptive score thresholds

- **Phần 4**: Reranking Improvements (2 proposals)
  - Two-stage reranking
  - Score calibration across topics

- **Plus**: Implementation roadmap, metrics, testing guide

**Khi nào đọc:** Khi implement từng improvement cụ thể

---

## 🗂️ CẤU TRÚC CODE MỚI

```
chatbot_v2/
│
├── 📁 app/                          # Application code (REFACTORED)
│   ├── api/                         # API Layer
│   │   ├── routes/                  # Endpoints
│   │   │   ├── chat.py             # /chat endpoint
│   │   │   └── health.py           # /health endpoint
│   │   └── dependencies.py         # Dependency injection
│   │
│   ├── core/                        # Core configuration
│   │   ├── config.py               # ⚙️ Centralized settings
│   │   ├── constants.py            # 📌 Application constants
│   │   ├── exceptions.py           # Custom exceptions
│   │   └── prompts/                # Prompt templates
│   │       ├── chat_prompts.py
│   │       └── evaluation_prompts.py
│   │
│   ├── domain/                      # Domain models
│   │   └── models.py               # 📦 Pydantic models
│   │
│   ├── services/                    # 🧠 Business Logic Layer
│   │   ├── llm_service.py          # LLM operations
│   │   ├── rag_service.py          # RAG orchestration (skeleton)
│   │   ├── chat_service.py         # Chat flow (skeleton)
│   │   └── reranker_service.py     # Reranking logic
│   │
│   ├── repositories/                # 💾 Data Access Layer
│   │   ├── chat_history_repo.py    # Database operations
│   │   ├── vector_store_repo.py    # Vector DB operations
│   │   └── bm25_repo.py            # BM25 retrieval
│   │
│   ├── adapters/                    # 🔌 External Services
│   │   ├── vllm_adapter.py         # vLLM API client
│   │   ├── qdrant_adapter.py       # Qdrant client
│   │   ├── xinference_adapter.py   # Reranker client
│   │   └── embedding_adapter.py    # Embedding model
│   │
│   └── utils/                       # 🛠️ Utilities
│       ├── logging.py              # Structured logging
│       ├── timing.py               # Performance timing
│       ├── text_processing.py      # Text utilities
│       └── validators.py           # Validation helpers
│
├── 📁 scripts/                      # Processing scripts (OLD)
│   ├── 1_document_splitting.py     # Chunking logic
│   ├── indexing_with_context.py    # Vector DB indexing
│   └── ...
│
├── 📄 main.py                       # 🚀 NEW entry point
├── 📄 app_new.py                    # OLD entry point (reference)
├── 📄 query_server_new.py           # OLD RAG server (reference)
│
├── 📘 REFACTOR_NOTES.md             # Architecture documentation
├── 📗 RAG_IMPROVEMENTS_PROPOSAL.md  # Implementation guide
└── 📖 README_IMPLEMENTATION.md      # This file
```

---

## 🎯 IMPLEMENTATION ROADMAP

### **Phase 1: Quick Wins** (1-2 tuần) ⭐ RECOMMENDED START HERE

| # | Improvement | Files to Create/Modify | Expected Gain |
|---|-------------|------------------------|---------------|
| 1 | **Adaptive Chunking** | Create: `app/services/chunking_service.py`<br>Modify: `scripts/1_document_splitting.py` | +15% precision |
| 2 | **Sparse Embeddings** | Modify: `app/adapters/qdrant_adapter.py`<br>Modify: `scripts/indexing_with_context.py` | +5-8% precision |
| 3 | **Query Classification** | Create: `app/services/query_classifier.py`<br>Create: `app/services/smart_retrieval_service.py` | +12% precision<br>-40% latency |
| 4 | **Topic-Adaptive Thresholds** | Modify: `app/core/constants.py`<br>Modify: `app/repositories/vector_store_repo.py` | +10% recall |

**Total Expected Gain:** +15-20% overall quality

---

### **Phase 2: Advanced** (2-3 tuần)

| # | Improvement | Files to Create/Modify | Expected Gain |
|---|-------------|------------------------|---------------|
| 5 | **Two-Stage Reranking** | Create: `app/services/two_stage_reranker.py`<br>Modify: `app/services/rag_service.py` | -60% reranking cost<br>+30% diversity |
| 6 | **Semantic Chunking** | Create: `app/services/semantic_chunking.py` | +5-10% precision |
| 7 | **Score Calibration** | Create: `app/services/score_calibration.py` | +20% cross-topic consistency |

**Total Expected Gain:** +10-15% additional quality

---

### **Phase 3: Scaling** (when data > 500K)

| # | Improvement | Complexity | When to Do |
|---|-------------|-----------|------------|
| 8 | Topic-Partitioned Collections | High | Only if data > 500K |

---

## 📝 CÁCH BẮT ĐẦU

### **Bước 1: Setup Environment** (10 phút)

```bash
# 1. Đọc documents
cat REFACTOR_NOTES.md
cat RAG_IMPROVEMENTS_PROPOSAL.md

# 2. Tạo branch mới cho improvement đầu tiên
git checkout -b feature/adaptive-chunking

# 3. Verify environment
python -c "from app.core.config import settings; print(settings.dict())"
```

---

### **Bước 2: Implement Improvement #1** (2-3 ngày)

**Target:** Adaptive Chunking

1. **Đọc proposal** (30 phút)
   ```bash
   # Đọc phần "PHẦN 1: CHUNKING IMPROVEMENTS"
   # Tìm section "Đề xuất 1: Adaptive Chunking theo Topic"
   ```

2. **Tạo service mới** (2 giờ)
   ```bash
   # Tạo file
   touch app/services/chunking_service.py

   # Copy code example từ proposal
   # Adjust cho codebase của bạn
   ```

3. **Define topic configs** (1 giờ)
   ```python
   # Trong app/core/constants.py
   TOPIC_CHUNKING_CONFIG = {
       "tcbc": {"chunk_size": 1536, "chunk_overlap": 384, "window_size": 1},
       "dvkh": {"chunk_size": 768, "chunk_overlap": 192, "window_size": 1},
       # ... 13 topics khác
   }
   ```

4. **Update chunking script** (3 giờ)
   ```python
   # Trong scripts/1_document_splitting.py
   from app.services.chunking_service import AdaptiveChunkingService

   chunker = AdaptiveChunkingService()
   chunks = chunker.create_chunks(content, topic, metadata)
   ```

5. **Test** (4 giờ)
   ```bash
   # Test với 2-3 topics trước
   python scripts/test_chunking.py --topics tcbc,dvkh,ktcn

   # So sánh old vs new chunking
   # Measure: avg chunk size, chunk distribution, content coverage
   ```

6. **Commit**
   ```bash
   git add .
   git commit -m "feat: Implement adaptive chunking by topic

   - Add AdaptiveChunkingService with topic-specific configs
   - Update document splitting to use adaptive strategy
   - Add chunking tests

   Expected: +15% precision
   "
   git push origin feature/adaptive-chunking
   ```

---

### **Bước 3: Measure Impact** (1 ngày)

```python
# Create test set
test_queries = [
    {"query": "Giá cước gửi hàng 5kg?", "topic": "bccp_nd", "ground_truth": ["doc123"]},
    # ... 50+ queries
]

# Compare old vs new
old_precision = evaluate_chunking(old_chunks, test_queries)
new_precision = evaluate_chunking(new_chunks, test_queries)

print(f"Improvement: {new_precision - old_precision:.1%}")
```

---

### **Bước 4: Repeat cho các improvements khác**

Lặp lại Bước 2-3 cho:
- Improvement #2: Sparse Embeddings
- Improvement #3: Query Classification
- v.v.

---

## 🆘 KHI CẦN HELP

### **Về Architecture:**
- ❓ "Service này nên inject dependency nào?"
- ❓ "Làm sao để test service có database dependency?"
- ❓ "Config này nên để ở đâu?"

### **Về RAG:**
- ❓ "Topic X nên config chunk_size bao nhiêu?"
- ❓ "Query này thuộc type gì?"
- ❓ "Threshold này quá cao hay thấp?"

### **Về Implementation:**
- ❓ "Code này có bug không?"
- ❓ "Làm sao optimize đoạn này?"
- ❓ "Error này nghĩa là gì?"

**→ Hãy hỏi tôi bất cứ lúc nào!**

---

## 📊 TRACKING PROGRESS

### **Checklist Implementation:**

#### Phase 1: Quick Wins
- [ ] Adaptive Chunking
  - [ ] Create `chunking_service.py`
  - [ ] Define `TOPIC_CHUNKING_CONFIG`
  - [ ] Update `1_document_splitting.py`
  - [ ] Test & measure

- [ ] Sparse Embeddings
  - [ ] Update `qdrant_adapter.py`
  - [ ] Modify indexing script
  - [ ] Re-index sample data
  - [ ] Test & measure

- [ ] Query Classification
  - [ ] Create `query_classifier.py`
  - [ ] Create `smart_retrieval_service.py`
  - [ ] Integrate to chat flow
  - [ ] Test & measure

- [ ] Topic-Adaptive Thresholds
  - [ ] Define `TOPIC_SCORE_THRESHOLDS`
  - [ ] Update retrieval logic
  - [ ] Test & measure

#### Phase 2: Advanced
- [ ] Two-Stage Reranking
- [ ] Semantic Chunking
- [ ] Score Calibration

---

## 🎓 LEARNING RESOURCES

### **Understanding the Code:**
```bash
# 1. Start với simple example
python -c "
from app.services.llm_service import LLMService
from app.adapters.vllm_adapter import VLLMAdapter

llm = LLMService(VLLMAdapter())
# Explore the code flow
"

# 2. Trace một API call
# Set breakpoint trong app/api/routes/chat.py
# Follow execution through services → repositories → adapters
```

### **Debugging:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use timing decorator
from app.utils.timing import timed

@timed
def my_function():
    # Function automatically logs execution time
    pass
```

---

## 🚀 QUICK COMMANDS

```bash
# Run refactored app
python main.py

# Run old app (for comparison)
python app_new.py

# Run tests (when you create them)
pytest tests/

# Check code quality
flake8 app/
black app/ --check

# Git workflow
git checkout -b feature/improvement-name
git add .
git commit -m "feat: description"
git push origin feature/improvement-name
```

---

## 📞 CONTACT

Khi implement, cứ hỏi tôi:
- Giải thích concept
- Review code
- Debug errors
- Optimize performance
- Design decisions

**Tôi sẵn sàng giúp bạn! 💪**

---

## ✅ READY TO START?

**Recommended first step:**
```bash
# 1. Đọc documents
cat REFACTOR_NOTES.md
cat RAG_IMPROVEMENTS_PROPOSAL.md

# 2. Bắt đầu với Adaptive Chunking
git checkout -b feature/adaptive-chunking

# 3. Ask me anything!
```

**Good luck! 🚀**
