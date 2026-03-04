import chromadb
from chromadb.config import Settings
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
DB_DIR = "chroma_db"


def get_client():
    """Create OpenAI client AFTER env vars are loaded."""
    return OpenAI()


def get_embedding(texts):
    client = get_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def build_and_save_index(chunks):
    chroma_client = chromadb.Client(Settings(persist_directory=DB_DIR))
    collection = chroma_client.get_or_create_collection(name="swiggy_report")

    embeddings = get_embedding(chunks)
    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    chroma_client.persist()
    print("ChromaDB index created.")
    

def load_collection():
    chroma_client = chromadb.Client(Settings(persist_directory=DB_DIR))
    return chroma_client.get_collection(name="swiggy_report")
