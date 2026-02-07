import os
from pathlib import Path
from dotenv import load_dotenv

import hashlib
import chromadb
from chromadb.config import Settings
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

load_dotenv()

def setup_chroma_index():
    # Create persistent client
    client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))

    # Create or get collection
    collection = client.get_or_create_collection(
        name="hcm_documents",
        metadata={"hnsw:space": "cosine"} # or "l2" for L2 distance
    )
    
    return client, collection

def setup_supabase_client() -> Client:
    supabase_url = os.getenv("PUBLIC_SUPABASE_URL")
    supabase_key = os.getenv("PUBLIC_SUPABASE_API")
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_API_KEY")
    return create_client(supabase_url, supabase_key)

def get_document_store():
    mode = os.getenv("DB_MODE", "local")

    if mode == "online":
        return setup_supabase_client()
    else:
        return setup_chroma_index()

def generate_id_from_text(chunk: str, metadata: dict = None) -> str:
    base = chunk
    if metadata:
        base += str(metadata.get("source", "")) + str(metadata.get("page", ""))
    return hashlib.sha256(base.encode()).hexdigest()

def add_to_chroma(collection, embeddings, text_chunks, all_metadata):
    seen_ids = set()
    unique_chunks = []
    unique_embeddings = []
    unique_metadata = []
    unique_ids = []
    skip_chunk = 0

    for i, chunk in enumerate(text_chunks):
        chunk_id = generate_id_from_text(chunk)
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

def get_existing_ids(supabase: Client) -> set:
    """Fetch all existing IDs from Supabase to prevent duplicate upserts."""
    response = supabase.table("hcm_documents").select("id").execute()
    if not response.data:
        return set()
    return set(item["id"] for item in response.data)

def add_to_supabase(supabase: Client, embeddings, text_chunks, metadata_list):
    # Load existing IDs once per script run
    existing_ids = get_existing_ids(supabase)

    records = []
    seen_ids = set()
    records = []
    skip_chunk = 0
    duplicate_chunk = 0

    for i, chunk in enumerate(text_chunks):
        chunk_id = generate_id_from_text(chunk, metadata_list[i])
        if chunk_id in existing_ids:
            skip_chunk += 1
            continue

        if chunk_id in seen_ids:
            duplicate_chunk += 1
            continue

        seen_ids.add(chunk_id)

        record = {
            "id": chunk_id,
            "content": chunk,
            "embedding": embeddings[i].tolist(),
            **metadata_list[i],
        }
        records.append(record)

    if records:
        response = supabase.table("hcm_documents").upsert(records).execute()
        print(f"Inserted {len(records)} records to Supabase")
    else:
        print("No records inserted — all chunks were duplicates.")

    print(f"Skipped {skip_chunk} chunks already in Supabase")
    print(f"Skipped {duplicate_chunk} duplicates within batch")

def search_supabase(supabase: Client, query_embedding, k=5):
    response = supabase.rpc("search_documents", {
        "query_embedding": query_embedding.tolist(),
        "top_k": k
    }).execute()

    if hasattr(response, "data"):
        return response.data
    else:
        raise RuntimeError(f"Supabase search failed: {response}")

def search_documents(query, model, store, k=5):
    query_embedding = model.encode([query])[0]
    if isinstance(store, tuple): # Local Chroma
        print("Searching in Chroma...")
        _, collection = store
        return search_chroma(collection, query_embedding, k)
    else: # Supabase
        print("Searching in Supabase...")
        return search_supabase(store, query_embedding, k)

def process_hcm_documents(docs_dir: Path, store, model):
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

        # add_to_chroma(collection, embeddings, text_chunks, chunk_metadata)
        if isinstance(store, tuple):
            _, collection = store
            add_to_chroma(collection, embeddings, text_chunks, chunk_metadata)
        else:
            add_to_supabase(store, embeddings, text_chunks, chunk_metadata)
    

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
    # client, collection = setup_chroma_db()
    
    # Process documents
    store = get_document_store()
    # process_hcm_documents(docs_dir, store, model)

    # print("\nImport complete!")
    # print(f"Total documents in collection: {collection.count()}")

    print("\nSample Query Results:")
    results = search_documents("human resource management", model, store, k=3)
    if isinstance(store, tuple):
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            print(f"Chapter: {metadata['chapter']}")
            print(f"Text: {doc[:200]}...")
            print("---")
    else:
        for row in results:
            print(f"Chapter: {row['chapter']}")
            print(f"Text: {row['content'][:200]}...")
            print("---")


if __name__ == "__main__":
    main()