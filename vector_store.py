import os
import logging
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArchaeologicalVectorStore:
    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL_NAME)
        self.vector_store: Optional[FAISS] = None

    def initialize_store(self, custom_file_path: Optional[str] = None, force_rebuild: bool = False) -> FAISS:
        source_path = custom_file_path or self.config.DEFAULT_PDF_PATH

        if not custom_file_path and os.path.exists(self.config.FAISS_INDEX_DIR) and not force_rebuild:
            logger.info("Loading pre-computed FAISS index...")
            self.vector_store = FAISS.load_local(
                folder_path=self.config.FAISS_INDEX_DIR,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            return self.vector_store

        if not os.path.exists(source_path):
            if os.path.exists(self.config.RAW_DATA_PATH):
                source_path = self.config.RAW_DATA_PATH
            else:
                raise FileNotFoundError(f"File not found: {source_path}")

        logger.info("Building FAISS index with metadata from %s...", source_path)
        if source_path.endswith(".pdf"):
            loader = PyPDFLoader(source_path)
            raw_docs = loader.load()
        else:
            loader = TextLoader(source_path, encoding="utf-8")
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.metadata["page"] = 1

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "PLATE", "CHAPTER", " "]
        )
        chunks = splitter.split_documents(raw_docs)

        self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
        if not custom_file_path:
            self.vector_store.save_local(self.config.FAISS_INDEX_DIR)

        return self.vector_store

    def get_retriever(self, top_k: Optional[int] = None):
        if not self.vector_store:
            self.initialize_store()
        k = top_k or self.config.RETRIEVER_TOP_K
        return self.vector_store.as_retriever(search_kwargs={"k": k})
