# Refactoring Notes

## Overview
This codebase has been refactored from a monolithic structure to a clean, layered architecture.

## Old Structure Problems
- `app_new.py`: 760 lines - mixed API routes, business logic, and database operations
- `query_server_new.py`: 1222 lines - everything related to RAG in one file
- Duplicate code (topic mappings, validation logic)
- Hard to test, hard to maintain
- No separation of concerns

## New Architecture

```
app/
├── api/              # API Layer - FastAPI routes
│   ├── routes/       # Endpoint definitions
│   └── dependencies. py # Dependency injection
│
├── core/             # Core configuration and constants
│   ├── config.py     # Centralized settings
│   ├── constants.py  # Application constants
│   ├── exceptions.py # Custom exceptions
│   └── prompts/      # Prompt templates
│
├── domain/           # Domain models
│   └── models.py     # Pydantic models
│
├── services/         # Business Logic Layer
│   ├── llm_service.py       # LLM interactions
│   ├── rag_service.py       # RAG orchestration
│   ├── chat_service.py      # Chat flow orchestration
│   └── reranker_service.py  # Reranking logic
│
├── repositories/     # Data Access Layer
│   ├── chat_history_repo.py # Database operations
│   ├── vector_store_repo.py # Vector DB operations
│   └── bm25_repo.py         # BM25 retrieval
│
├── adapters/         # External Service Adapters
│   ├── vllm_adapter.py       # vLLM API client
│   ├── qdrant_adapter.py     # Qdrant client
│   ├── xinference_adapter.py # Xinference reranker
│   └── embedding_adapter.py  # Embedding model
│
└── utils/            # Utilities
    ├── logging.py    # Structured logging
    ├── timing.py     # Performance timing
    ├── text_processing.py # Text utils
    └── validators.py # Validation helpers
```

## Benefits

### 1. Separation of Concerns
- API layer only handles HTTP requests/responses
- Services contain business logic
- Repositories handle data access
- Adapters wrap external services

### 2. Testability
- Each layer can be tested independently
- Mock dependencies easily
- Unit tests without real databases/APIs

### 3. Maintainability
- Small, focused files (< 200 lines each)
- Clear responsibilities
- Easy to locate code

### 4. Flexibility
- Swap implementations easily (e.g., vLLM -> OpenAI)
- Add new features without touching existing code
- Configuration centralized

### 5. Dependency Injection
- Dependencies declared explicitly
- No global state
- FastAPI handles lifecycle

## Migration Status

### ✅ Completed (Phase 1)
- [x] Core configuration and constants
- [x] Domain models
- [x] All adapters (vLLM, Qdrant, Xinference, Embedding)
- [x] All repositories
- [x] Service layer structure (LLM, Reranker, RAG, Chat)
- [x] API routes structure
- [x] Dependency injection setup
- [x] New main.py entry point

### 🚧 In Progress (Phase 2)
- [ ] Complete RAGService implementation (retrieve_and_process)
- [ ] Complete ChatService implementation (full chat flow)
- [ ] Context builder logic
- [ ] Evaluation service
- [ ] RAG API routes (rag_query endpoint)
- [ ] Migration from old files

### ⏳ TODO (Phase 3)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Documentation updates
- [ ] Remove old files (app_new.py, query_server_new.py)

## How to Run

### Current (Refactored):
```bash
# Make sure .env is configured
python main.py
```

### Old (Before refactoring):
```bash
python app_new.py  # Still available for comparison
```

## Next Steps

1. **Complete RAG Service**: Migrate full retrieval logic from `query_server_new.py:retrieve_and_process`
2. **Complete Chat Service**: Migrate full chat flow from `app_new.py:chat_endpoint`
3. **Add remaining endpoints**: `/rag_query`, debug endpoints
4. **Testing**: Write unit tests for each layer
5. **Deprecate old files**: Once everything works, remove `app_new.py` and `query_server_new.py`

## Code Examples

### Before (Monolithic):
```python
# app_new.py - 178 lines in one function!
@app.post("/chat")
async def chat_endpoint(...):
    # validation
    # topic normalization
    # vLLM health check
    # query rewriting
    # RAG call
    # response generation
    # metadata evaluation
    # database save
    # error handling
    ...
```

### After (Clean):
```python
# api/routes/chat.py - 20 lines
@router.post("/chat")
async def chat_endpoint(request, service=Depends(...)):
    return await service.process_chat(request)

# services/chat_service.py - orchestrates
class ChatService:
    async def process_chat(self, request):
        standalone_q = await self.llm.rewrite_query(...)
        context = await self.rag.retrieve(...)
        response = await self.llm.generate_response(...)
        # ...
```

## Questions?

Contact the development team or refer to:
- Architecture Decision Records (ADRs) - coming soon
- API documentation: http://localhost:6868/docs
