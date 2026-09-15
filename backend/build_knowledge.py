import json
import os
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "website_content.json"
)

CHROMA_DIR = os.path.join(
    os.path.dirname(__file__),
    "data",
    "chroma_db"
)

COLLECTION_NAME = "flux_website"


# ============================================================
# CHUNK TEXT
# ============================================================

def create_chunks(text, chunk_size=800, overlap=150):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# LOAD WEBSITE DATA
# ============================================================

def load_website_data():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Could not find {DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def main():

    print("=" * 60)
    print("FLUX RAG KNOWLEDGE BASE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load website pages
    # --------------------------------------------------------

    pages = load_website_data()

    print()
    print("Website pages loaded:", len(pages))

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    all_chunks = []

    for page in pages:

        chunks = create_chunks(
            page["content"]
        )

        print(
            f'{page["url"]} → {len(chunks)} chunks'
        )

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "url": page["url"],
                "title": page["title"],
                "chunk_id": index
            })

    print()
    print("Total chunks:", len(all_chunks))

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # Create ChromaDB
    # --------------------------------------------------------

    print()
    print("Creating ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # Delete old collection if it exists
    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print("Old collection deleted.")

    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    texts = [
        item["text"]
        for item in all_chunks
    ]

    print()
    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    # --------------------------------------------------------
    # Add to ChromaDB
    # --------------------------------------------------------

    ids = [
        f"chunk_{i}"
        for i in range(len(all_chunks))
    ]

    documents = texts

    metadatas = [
        {
            "url": item["url"],
            "title": item["title"],
            "chunk_id": item["chunk_id"]
        }
        for item in all_chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("KNOWLEDGE BASE READY")
    print("=" * 60)

    print(
        "Documents stored:",
        collection.count()
    )

    print(
        "Database:",
        CHROMA_DIR
    )

    print("=" * 60)


if __name__ == "__main__":
    main()