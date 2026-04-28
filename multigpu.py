import os
import pandas as pd
import numpy as np
import kagglehub
import chromadb
import torch
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. Dataset Prep & Chunking
# ==========================================
print("--- Phase 1: Preparing Data ---")
dataset_dir = kagglehub.dataset_download("rtatman/ubuntu-dialogue-corpus")
csv_path = os.path.join(dataset_dir, "Ubuntu-dialogue-corpus", "dialogueText.csv")

df = pd.read_csv(csv_path).dropna(subset=['text'])
sample_ids = df['dialogueID'].drop_duplicates().sample(n=5000, random_state=42)
df_dev = df[df['dialogueID'].isin(sample_ids)].copy()
df_dev['formatted_text'] = df_dev['from'].astype(str) + ": " + df_dev['text'].astype(str)
df_dev = df_dev.sort_values(by=['dialogueID', 'date'])

conversations = df_dev.groupby('dialogueID').agg(
    full_text=('formatted_text', lambda x: '\n'.join(x)),
    folder=('folder', 'first')
).reset_index()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200, # Increased chunk size since we have a better LLM
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

final_chunks, final_metadata = [], []
for index, row in conversations.iterrows():
    chunks = text_splitter.split_text(row['full_text'])
    for i, chunk in enumerate(chunks):
        final_chunks.append(chunk)
        final_metadata.append({"dialogueID": row['dialogueID'], "chunk_index": i})

# ==========================================
# 2. Advanced Vectorization (BGE-Large)
# ==========================================
print("\n--- Phase 2: Vectorization ---")
# BGE-Large is much more accurate but computationally heavier. Perfect for your GPUs.
embed_model_name = 'BAAI/bge-large-en-v1.5'
# normalized_embeddings=True is strictly required for BGE models to use cosine similarity correctly
bi_encoder = SentenceTransformer(embed_model_name, model_kwargs={"torch_dtype": torch.float16})

chroma_client = chromadb.PersistentClient(path="./chroma_db_bge")
collection = chroma_client.get_or_create_collection(
    name="ubuntu_bge_large",
    metadata={"hnsw:space": "cosine"}
)

if collection.count() == 0:
    ids = [f"chunk_{i}" for i in range(len(final_chunks))]
    batch_size = 250 # Reduced batch size for the larger embedding model
    
    print(f"Generating embeddings for {len(final_chunks)} chunks...")
    for i in tqdm(range(0, len(final_chunks), batch_size)):
        batch_chunks = final_chunks[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_metadata = final_metadata[i : i + batch_size]
        
        # SentenceTransformers will automatically utilize your GPUs
        batch_embeddings = bi_encoder.encode(batch_chunks, normalize_embeddings=True).tolist()
        
        collection.add(
            ids=batch_ids, documents=batch_chunks, embeddings=batch_embeddings, metadatas=batch_metadata
        )
else:
    print(f"Skipping indexing: Found {collection.count()} chunks already in database.")

# ==========================================
# 3. Load Re-Ranker & 14B LLM (Multi-GPU)
# ==========================================
print("\n--- Phase 3: Loading Heavy Models ---")
# Advanced Re-ranker
cross_encoder = CrossEncoder('BAAI/bge-reranker-large', default_activation_function=torch.nn.Sigmoid())

# 14 Billion Parameter LLM
llm_model_id = "Qwen/Qwen2.5-14B-Instruct"
print(f"Loading {llm_model_id} across 2x A5000s...")

tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.bfloat16, # Ampere architecture (A5000) highly optimized for bfloat16
    device_map="auto"           # MAGIC HAPPENS HERE: Automatically splits the model across both GPUs!
)

# ==========================================
# 4. Core RAG Function
# ==========================================
def ask_ubuntu_rag(query_text, top_k=3):
    """Executes the full Retrieval, Re-ranking, and Generation pipeline."""
    
    # 1. Retrieval
    query_vector = bi_encoder.encode(query_text, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=30)
    
    retrieved_documents = results['documents'][0]
    
    # 2. Re-ranking
    cross_input_pairs = [[query_text, doc] for doc in retrieved_documents]
    cross_scores = cross_encoder.predict(cross_input_pairs)
    sorted_indices = np.argsort(cross_scores)[::-1]
    
    # Extract Context
    top_docs = [retrieved_documents[sorted_indices[i]] for i in range(top_k)]
    context_block = "\n---\n".join(top_docs)
    
    # 3. Generation Prompt
    messages = [
        {"role": "system", "content": "You are a highly technical, factual Ubuntu support AI. Answer the user's question using ONLY the provided context. If the context does not contain the answer, reply exactly with: 'I cannot find the answer in the provided logs.' Do not use outside knowledge."},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query_text}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(llm.device)
    
    # 4. Generation
    outputs = llm.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,    # Low temp for accuracy
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    input_length = inputs.input_ids.shape[1]
    final_answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    
    return final_answer

# ==========================================
# 5. Automated Validation & Stress Testing
# ==========================================
print("\n--- Phase 4: Stress Testing the Pipeline ---")

test_cases = [
    {
        "name": "Standard Retrieval Test",
        "query": "How do I extract a .tar.gz file?",
        "expected_behavior": "Should mention 'tar -zxvf' or similar tar commands."
    },
    {
        "name": "Hallucination Resistance Test (Out of Bounds)",
        "query": "How do I install Windows 11 on a Macbook Pro?",
        "expected_behavior": "Should state it cannot find the answer in the logs."
    },
    {
        "name": "Semantic Vague Troubleshooting Test",
        "query": "My screen looks completely weird and the graphics are messed up.",
        "expected_behavior": "Should retrieve logs related to xorg, drivers, display resolution, or graphics."
    }
]

print("="*60)
for test in test_cases:
    print(f"🧪 TEST: {test['name']}")
    print(f"❓ QUERY: {test['query']}")
    print(f"🎯 EXPECTED: {test['expected_behavior']}")
    
    answer = ask_ubuntu_rag(test['query'])
    
    print(f"🤖 AI OUTPUT:\n{answer}")
    print("="*60)