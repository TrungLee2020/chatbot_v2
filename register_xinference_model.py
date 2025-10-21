import json
from xinference.client import Client
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()

# Huggingface login
hf_key = os.getenv("HF_API_KEY")
if hf_key:
    login(hf_key)

endpoint = os.environ.get("XINFERENCE_ENDPOINT")
client = Client(endpoint)
rerank_model_name = os.environ.get("RERANK_MODEL_NAME", "custom-gte-multilingual-reranker")

# Config đúng chuẩn theo documentation
reranker_config = {
    "model_name": rerank_model_name,
    "type": "normal",
    "language": ["en", "vi", "multilingual"],
    "model_id": "Alibaba-NLP/gte-multilingual-reranker-base",
    "model_uri": "file:///root/.xinference/models/gte-multilingual-reranker-base"
}

try:
    # Kiểm tra registered models
    registered_models = client.list_model_registrations("rerank")
    print(f"Currently registered rerank models: {[model['model_name'] for model in registered_models]}")
    
    # Unregister model cũ nếu có
    if rerank_model_name in [model['model_name'] for model in registered_models]:
        print(f"Unregistering existing model: {rerank_model_name}")
        client.unregister_model(model_type="rerank", model_name=rerank_model_name)
        print("Model unregistered successfully.")
    
    # Register model với config đúng chuẩn
    print("Registering model with correct config format...")
    client.register_model(
        model_type="rerank", 
        model=json.dumps(reranker_config), 
        persist=True
    )
    print("✅ Custom reranker model registered successfully with correct format.")
    
    # Verify registration
    registered_models = client.list_model_registrations("rerank")
    print(f"Updated registered rerank models: {[model['model_name'] for model in registered_models]}")
    
    # In ra config để verify
    print(f"\nRegistered config: {json.dumps(reranker_config, indent=2)}")
    
except Exception as e:
    print(f"❌ An error occurred: {e}")
    import traceback
    traceback.print_exc()