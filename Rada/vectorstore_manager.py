# vectorstore_manager.py
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

class VectorStoreManager:
    def __init__(self, persist_directory="db/chroma"):
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory

        #self.embedding_fn = OpenAIEmbeddings()

        # AzureOpenAIEmbeddings 
        AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        self.embedding_fn = AzureOpenAIEmbeddings(
            deployment=AZURE_OPENAI_DEPLOYMENT,
            model=AZURE_OPENAI_DEPLOYMENT,
            azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_fn
        )
        
    def create_new_session_store(self, session_id: str):
        session_path = os.path.join(self.persist_directory, session_id)
        return Chroma(
            persist_directory=session_path,
            embedding_function=self.embedding_fn
        )
    def add_chunks(self, chunks_with_metadata):
        docs = []
        for text, metadata in chunks_with_metadata:
            clean_md = {}
            pdf_info = metadata.get("pdf_info", {})
            
            # Estrazione metadati base
            clean_md["filename"] = pdf_info.get("filename", "")
            clean_md["pdf_index"] = pdf_info.get("pdf_index", 0)
            clean_md["page"] = metadata.get("page", 1)
            clean_md["chunk_id"] = metadata.get("chunk_id", "")
            clean_md["session_id"] = metadata.get("session_id", "")  # <-- AGGIUNTA CRUCIALE

            docs.append(Document(page_content=text, metadata=clean_md))

        if docs:
            self.vectordb.add_documents(docs)
            self.vectordb.persist()
        return self.vectordb

    def get_retriever(self, **kwargs):
        return self.vectordb.as_retriever(**kwargs)

    def clear(self):
        self.vectordb.delete_collection()
        self.vectordb.persist()