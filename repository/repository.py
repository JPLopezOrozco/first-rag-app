import chromadb 
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os



load_dotenv()
embedding_fn = embedding_functions.VoyageAIEmbeddingFunction(
    api_key=os.environ["VOYAGE_API_KEY"],
    model_name="voyage-4-lite",
)
client = chromadb.PersistentClient(path="chromadb")

collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(filename: str, chunks: list[str], metadata: list[dict], ids: list[str]) -> None:
    collection.delete(where={"source": filename})
    collection.upsert(documents=chunks, metadatas=metadata, ids=ids)


def search(query: str, n_results: int = 4) -> dict:
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )