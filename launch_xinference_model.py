from xinference.client import Client
from dotenv import load_dotenv
import os
import time

load_dotenv()

endpoint = os.environ.get("XINFERENCE_ENDPOINT")
client = Client(endpoint)
rerank_model_name = os.environ.get("RERANK_MODEL_NAME", "custom-gte-multilingual-reranker")

try:
    print("🚀 Launching reranker model...")
    
    # Launch model đơn giản theo documentation
    model_uid = client.launch_model(model_name=rerank_model_name,
                model_type="rerank",
                model_format="pytorch",
                model_size_in_billions=0.3,
                quantization="none"
            )
    print(f"✅ Model launched successfully. UID: {model_uid}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()