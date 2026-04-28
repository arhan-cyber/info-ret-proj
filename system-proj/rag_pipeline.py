# rag_pipeline.py
import torch
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import config

class RAGPipeline:
    def __init__(self, vector_store):
        print("\n--- Phase 3: Loading Heavy Models ---")
        self.vector_store = vector_store
        
        # Load Re-ranker
        self.cross_encoder = CrossEncoder(
            config.RERANKER_MODEL_NAME, 
            default_activation_function=torch.nn.Sigmoid()
        )

        # Load LLM
        print(f"Loading {config.LLM_MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_ID,
            torch_dtype=config.LLM_DTYPE,
            device_map="auto"
        )

    def ask(self, query_text, top_k=3):
        # 1. Retrieval
        retrieved_documents = self.vector_store.retrieve(query_text, top_n=30)
        
        # 2. Re-ranking
        cross_input_pairs = [[query_text, doc] for doc in retrieved_documents]
        cross_scores = self.cross_encoder.predict(cross_input_pairs)
        sorted_indices = np.argsort(cross_scores)[::-1]
        
        top_docs = [retrieved_documents[sorted_indices[i]] for i in range(top_k)]
        context_block = "\n---\n".join(top_docs)
        
        # 3. Generation Prompt
        messages = [
            {"role": "system", "content": "You are a highly technical, factual Ubuntu support AI. Answer the user's question using ONLY the provided context. If the context does not contain the answer, reply exactly with: 'I cannot find the answer in the provided logs.' Do not use outside knowledge."},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query_text}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.llm.device)
        
        # 4. Generation
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        input_length = inputs.input_ids.shape[1]
        final_answer = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
        
        return final_answer