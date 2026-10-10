from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    DEFAULT_PDF_PATH: str = "book_of_the_dead.pdf"
    RAW_DATA_PATH: str = "/content/book_of_the_dead.txt"
    FAISS_INDEX_DIR: str = "faiss_egyptian_index"
    UPLOAD_DIR: str = "uploaded_docs"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 120
    RETRIEVER_TOP_K: int = 3
    GEMINI_MODEL_NAME: str = "gemini-3.8-flash"
    DEFAULT_TEMPERATURE: float = 0.1
    LOCAL_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
    LOCAL_MAX_NEW_TOKENS: int = 512
