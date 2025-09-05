from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

def load_documents(documentList,CollectionName,ChunkSize=1000,ChunkOverlap=200):
    text_splitter = CharacterTextSplitter(chunk_size=ChunkSize, chunk_overlap=ChunkOverlap)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "./chroma_db"
    for doc in documentList:
        text=""
        print("Document {} loading started!!".format(doc))
        with open(doc,mode="r") as f:
            for line in f:
                text += line
        chunked_data = text_splitter.create_documents([text])
        vectordb = Chroma.from_documents(
            documents=chunked_data,
            collection_name=CollectionName,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectordb.persist()
        print("Document {} loaded!".format(doc))



