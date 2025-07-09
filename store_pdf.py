import numpy as np
import pickle
from pathlib import Path

import chromadb
from chromadb.config import Settings
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
from sentence_transformers import SentenceTransformer

# Load PDF
folder_path = Path("hcm_files")
embedding_model_name = "all-MiniLM-L6-v2"
token_chunk_size = 40
token_chunk_overlap = 0
# pdf_path = Path("hcm_files", "hcm_chap15_1014_1141.pdf")

"""
Chap1: 147-180
Chap15: 1048-1141
"""

def setup_chroma_index():
    # Create persistent client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="hcm_documents",
        metadata={"hnsw:space": "cosine"}  # or "l2" for L2 distance
    )
    
    return client, collection

def add_to_chroma(collection, embeddings, text_chunks, all_metadata):
    # Chroma expects embeddings as list of lists
    embeddings_list = embeddings.tolist()
    
    # Create unique IDs
    ids = [f"chunk_{i}" for i in range(len(text_chunks))]
    
    # Add to collection
    collection.add(
        embeddings=embeddings_list,
        documents=text_chunks,
        metadatas=all_metadata,
        ids=ids
    )

def search_chroma(collection, query_embedding, k=5):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    return results

# Load embedding model
model = SentenceTransformer(embedding_model_name) # fast and good enough
# dimension = model.get_sentence_embedding_dimension()
# index = faiss.IndexFlatL2(dimension) # L2 distance index
# Setup Chroma
client, collection = setup_chroma_index()

all_chunks = []
all_metadata = []

# Process each PDF
for pdf_file in folder_path.glob("*.pdf"):
    chapter_name = pdf_file.stem
    print(f"Processing {chapter_name}...")

    # doc = pymupdf.open(pdf_file)
    md_text = pymupdf4llm.to_markdown(str(pdf_file))

    # Split Markdown into token-based chunks
    splitter = MarkdownTextSplitter(chunk_size=token_chunk_size, chunk_overlap=token_chunk_overlap)
    documents = splitter.create_documents([md_text])
    text_chunks = [doc.page_content for doc in documents]
    print(f"{chapter_name}: {len(text_chunks)} chunks")

    # Embed chunks
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # buffer = ""
    # text_chunks = []

    # for page in doc:
    #     buffer += page.get_text()

    #     while len(buffer) > chunk_size:
    #         text_chunks.append(buffer[:chunk_size])
    #         buffer = buffer[chunk_size:]

    # if buffer:
    #     text_chunks.append(buffer)

    # Add to Chroma
    chunk_metadata = [{"chapter": chapter_name} for _ in text_chunks]
    add_to_chroma(collection, embeddings, text_chunks, chunk_metadata)

    all_chunks.extend(text_chunks)
    all_metadata.extend({"chapter": chapter_name, "text": chunk} for chunk in text_chunks)


def search_documents(query, k=5):
    query_embedding = model.encode([query])[0]
    results = search_chroma(collection, query_embedding, k)
    return results

results = search_documents("human resource management", k=3)
for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
    print(f"Chapter: {metadata['chapter']}")
    print(f"Text: {doc[:200]}...")
    print("---")