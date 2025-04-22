import numpy as np
import pickle
from pathlib import Path

import faiss
import pymupdf
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

# Load embedding model
model = SentenceTransformer(embedding_model_name) # fast and good enough
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension) # L2 distance index

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

    index.add(embeddings)
    all_chunks.extend(text_chunks)
    all_metadata.extend({"chapter": chapter_name, "text": chunk} for chunk in text_chunks)

# Save the index and chunks
faiss.write_index(index, "hcm_index.faiss")

# Save the corresponding texts
with open("hcm_chunks.pkl", "wb") as f:
    pickle.dump(text_chunks, f)

with open("hcm_metadata.pkl", "wb") as f:
    pickle.dump(all_metadata, f)

print(f"Total Chunks: {len(all_chunks)}")