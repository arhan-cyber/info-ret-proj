# config.py
import torch

# Data prep settings
SAMPLE_SIZE = 5000
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Model settings
EMBED_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-large'
LLM_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# Database settings
CHROMA_PATH = "./chroma_db_bge"
COLLECTION_NAME = "ubuntu_bge_large"
BATCH_SIZE = 250

# Hardware
LLM_DTYPE = torch.bfloat16