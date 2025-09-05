import chromadb
from chromadb.config import Settings

def createdbNCollection(CollectionName):
    settings = Settings()
    print("*"*25,"Initializing ChromadDB","*"*25)
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=CollectionName,
        metadata={"hnsw:space": "cosine"}
    )
    print(f" Collection created: {collection.name}")
    print(f" Memories: {collection.count()}")
    print(" ChromaDB setup ready!")