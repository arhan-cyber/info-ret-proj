# vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config
import torch

class VectorStoreManager:
    def __init__(self):
        self.bi_encoder = SentenceTransformer(
            config.EMBED_MODEL_NAME, 
            model_kwargs={"torch_dtype": torch.float16}
        )
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def populate_database(self, final_chunks, final_metadata):
        print("\n--- Phase 2: Vectorization ---")
        if self.collection.count() > 0:
            print(f"Skipping indexing: Found {self.collection.count()} chunks already in database.")
            return

        ids = [f"chunk_{i}" for i in range(len(final_chunks))]
        print(f"Generating embeddings for {len(final_chunks)} chunks...")
        
        for i in tqdm(range(0, len(final_chunks), config.BATCH_SIZE)):
            batch_chunks = final_chunks[i : i + config.BATCH_SIZE]
            batch_ids = ids[i : i + config.BATCH_SIZE]
            batch_metadata = final_metadata[i : i + config.BATCH_SIZE]
            
            batch_embeddings = self.bi_encoder.encode(batch_chunks, normalize_embeddings=True).tolist()
            
            self.collection.add(
                ids=batch_ids, documents=batch_chunks, 
                embeddings=batch_embeddings, metadatas=batch_metadata
            )

    def retrieve(self, query_text, top_n=30):
        query_vector = self.bi_encoder.encode(query_text, normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=[query_vector], n_results=top_n)
        return results['documents'][0]