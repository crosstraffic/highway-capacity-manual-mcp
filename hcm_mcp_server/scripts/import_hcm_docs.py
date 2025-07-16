from pathlib import Path

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

def search_documents(query, model, collection, k=5):
    query_embedding = model.encode([query])[0]
    results = search_chroma(collection, query_embedding, k)
    return results

def process_hcm_documents(docs_dir: Path, collection, model):
    """Process HCM documents and add to collection."""
    doc_count = 0
    token_chunk_size = 40
    token_chunk_overlap = 0
    # Process each PDF
    for pdf_file in docs_dir.glob("*.pdf"):
        chapter_name = pdf_file.stem.replace('chapter', '')
        print(f"Processing {chapter_name}...")

        # doc = pymupdf.open(pdf_file)
        md_text = pymupdf4llm.to_markdown(str(pdf_file))

        # Split Markdown into token-based chunks
        splitter = MarkdownTextSplitter(chunk_size=token_chunk_size, chunk_overlap=token_chunk_overlap)
        documents = splitter.create_documents([md_text])
        text_chunks = [doc.page_content for doc in documents]
        print(f"{chapter_name}: {len(text_chunks)} chunks")

        embeddings = model.encode(text_chunks, show_progress_bar=True)
        chunk_metadata = [{"chapter": chapter_name, "source_file": pdf_file.name} for _ in text_chunks]
        add_to_chroma(collection, embeddings, text_chunks, chunk_metadata)
    
    print(f"Processed {doc_count} document chunks")


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
    
    print(f"\nImport complete!")
    print(f"Total documents in collection: {collection.count()}")

    results = search_documents("human resource management", model, collection, k=3)
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        print(f"Chapter: {metadata['chapter']}")
        print(f"Text: {doc[:200]}...")
        print("---")


if __name__ == "__main__":
    main()