from pathlib import Path

import hashlib
import chromadb
from chromadb.config import Settings
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
from sentence_transformers import SentenceTransformer

def setup_chroma_index():
    # Create persistent client
    client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))

    # Create or get collection
    collection = client.get_or_create_collection(
        name="hcm_documents",
        metadata={"hnsw:space": "cosine"} # or "l2" for L2 distance
    )
    
    return client, collection

def generate_id_from_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def add_to_chroma(collection, embeddings, text_chunks, all_metadata):
    seen_ids = set()
    unique_chunks = []
    unique_embeddings = []
    unique_metadata = []
    unique_ids = []
    skip_chunk = 0

    for i, chunk in enumerate(text_chunks):
        chunk_id = hashlib.md5(chunk.encode("utf-8")).hexdigest()
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)
            unique_embeddings.append(embeddings[i])
            unique_metadata.append(all_metadata[i])
            unique_ids.append(chunk_id)
        else:
            skip_chunk += 1

    # Add to collection
    collection.upsert(
        embeddings=unique_embeddings,
        documents=unique_chunks,
        metadatas=unique_metadata,
        ids=unique_ids
    )

    print(f"The number of skipping duplicate chunk: {skip_chunk}")

def search_chroma(collection, query_embedding, k=5):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )
    return results

def search_documents(query, model, collection, k=5):
    query_embedding = model.encode([query])[0]
    results = search_chroma(collection, query_embedding, k)
    return results

def process_hcm_documents(docs_dir: Path, collection, model):
    """Process HCM documents and add to collection."""
    token_chunk_size = 128
    token_chunk_overlap = 0

    # Process each PDF
    for pdf_file in docs_dir.glob("*.pdf"):
        chapter_name = pdf_file.stem.replace('chapter', '')
        print(f"Processing {chapter_name}...")

        md_text = pymupdf4llm.to_markdown(str(pdf_file))

        print(f"Markdown text length: {len(md_text)} characters")
        # Split Markdown into token-based chunks
        splitter = MarkdownTextSplitter(chunk_size=token_chunk_size, chunk_overlap=token_chunk_overlap)
        documents = splitter.create_documents([md_text])
        text_chunks = [doc.page_content for doc in documents]
        print(f"{chapter_name}: {len(text_chunks)} chunks")

        embeddings = model.encode(text_chunks, show_progress_bar=True)
        chunk_metadata = [{"chapter": chapter_name, "source_file": pdf_file.name} for _ in text_chunks]
        add_to_chroma(collection, embeddings, text_chunks, chunk_metadata)
    

def main():
    print("HCM Document Import")
    print("=" * 25)
    
    # Load PDF
    embedding_model_name = "all-MiniLM-L6-v2"

    # Setup
    docs_dir = Path("data/hcm_files")
    if not docs_dir.exists():
        print(f"ERROR: Documents directory not found: {docs_dir}")
        return
    
    # Initialize model
    print("Loading embedding model...")
    model = SentenceTransformer(embedding_model_name) # fast and good enough
    
    # Setup Chroma
    client, collection = setup_chroma_index()
    
    # Process documents
    process_hcm_documents(docs_dir, collection, model)
    
    print("\nImport complete!")
    print(f"Total documents in collection: {collection.count()}")

    results = search_documents("human resource management", model, collection, k=3)
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        print(f"Chapter: {metadata['chapter']}")
        print(f"Text: {doc[:200]}...")
        print("---")


if __name__ == "__main__":
    main()