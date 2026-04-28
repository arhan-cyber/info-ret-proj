# data_prep.py
import os
import pandas as pd
import kagglehub
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

def load_and_chunk_data():
    print("--- Phase 1: Preparing Data ---")
    dataset_dir = kagglehub.dataset_download("rtatman/ubuntu-dialogue-corpus")
    csv_path = os.path.join(dataset_dir, "Ubuntu-dialogue-corpus", "dialogueText.csv")

    df = pd.read_csv(csv_path).dropna(subset=['text'])
    sample_ids = df['dialogueID'].drop_duplicates().sample(n=config.SAMPLE_SIZE, random_state=42)
    df_dev = df[df['dialogueID'].isin(sample_ids)].copy()
    df_dev['formatted_text'] = df_dev['from'].astype(str) + ": " + df_dev['text'].astype(str)
    df_dev = df_dev.sort_values(by=['dialogueID', 'date'])

    conversations = df_dev.groupby('dialogueID').agg(
        full_text=('formatted_text', lambda x: '\n'.join(x)),
        folder=('folder', 'first')
    ).reset_index()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    final_chunks, final_metadata = [], []
    for index, row in conversations.iterrows():
        chunks = text_splitter.split_text(row['full_text'])
        for i, chunk in enumerate(chunks):
            final_chunks.append(chunk)
            final_metadata.append({"dialogueID": row['dialogueID'], "chunk_index": i})
            
    return final_chunks, final_metadata