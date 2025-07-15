from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def setup_chroma_db(db_path: str = "./chroma_db"):
    """Initialize ChromaDB with HCM collection."""
    print(f"Setting up ChromaDB at {db_path}")
    
    # Create client
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    
    # Create collection
    try:
        collection = client.create_collection(
            name="hcm_documents",
            metadata={"description": "Highway Capacity Manual documents and sections"}
        )
        print("Created HCM documents collection")
    except Exception as e:
        print(f"Collection may already exist: {e}")
        collection = client.get_collection(name="hcm_documents")
    
    return client, collection


def load_sample_hcm_data(collection, model):
    """Load sample HCM data for testing."""
    sample_documents = [
        {
            "content": "Two-lane highways are defined as roadways with one lane for each direction of travel...",
            "metadata": {"chapter": "15", "section": "Introduction", "page": "15-1"}
        },
        {
            "content": "Free-flow speed is the theoretical speed that would occur with low traffic volumes...",
            "metadata": {"chapter": "15", "section": "Free Flow Speed", "page": "15-8"}
        },
        {
            "content": "Level of service represents the quality of traffic flow and is designated by letters A through F...",
            "metadata": {"chapter": "15", "section": "Level of Service", "page": "15-25"}
        }
    ]
    
    documents = [doc["content"] for doc in sample_documents]
    metadatas = [doc["metadata"] for doc in sample_documents]
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    # Generate embeddings
    embeddings = model.encode(documents)
    
    # Add to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings.tolist()
    )
    
    print(f"Added {len(documents)} sample documents to collection")


def main():
    """Main setup function."""
    print("HCM-LLM Database Setup")
    print("=" * 30)
    
    # Initialize embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Setup database
    client, collection = setup_chroma_db()
    
    # Load sample data
    load_sample_hcm_data(collection, model)
    
    print("\nSetup complete!")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()

