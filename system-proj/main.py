# main.py
from data_prep import load_and_chunk_data
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline

def run_stress_tests(rag_engine):
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
        
        answer = rag_engine.ask(test['query'])
        
        print(f"🤖 AI OUTPUT:\n{answer}")
        print("="*60)

if __name__ == "__main__":
    # 1. Prep data
    chunks, metadata = load_and_chunk_data()
    
    # 2. Setup Vector Store & Index
    db_manager = VectorStoreManager()
    db_manager.populate_database(chunks, metadata)
    
    # 3. Init Heavy Models & Pipeline
    rag_engine = RAGPipeline(vector_store=db_manager)
    
    # 4. Run Tests
    run_stress_tests(rag_engine)