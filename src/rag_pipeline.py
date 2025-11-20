import os
import glob
import hashlib
import json
from dataclasses import dataclass
from src.interfaces.base_documents import BaseDocument
from src.interfaces.base_embedding import BaseEmbedding
from src.interfaces.base_retriever import BaseRetriever
from langchain_core.documents import Document
from src.implements.chunk_store_duck_db import DuckDBChunkStore

@dataclass
class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""
    document_processing: BaseDocument
    embedding_execution: BaseEmbedding
    # retriever: BaseRetriever
    
    METADATA_PATH = "db_agriculture3/processed_files.json"
    FILES_DIR = "/docs"
    EMBEDDING_VECTOR_DB_DIR = "db_agriculture3/chroma_db"
    EXPORT_CHUNKS_FILENAME = "chunks_export.json"
    CHUNK_DB_PATH = "chunks.duckdb"

    def __post_init__(self):
        self.chunk_store = DuckDBChunkStore(self.CHUNK_DB_PATH)

    def get_file_hash(self, file_path):
        """Tạo hash MD5 cho file để phát hiện thay đổi nội dung."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()


    def run_complete_ingestion_pipeline(self, pdf_dir: str, meta_path = METADATA_PATH):
        """Chỉ xử lý file PDF mới hoặc thay đổi."""
        print(" Starting Incremental RAG Ingestion Pipeline")
        print("=" * 50)

        # Load metadata cũ
        processed_meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                processed_meta = json.load(f)

        # Tìm tất cả file PDF
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        new_files = []

        for pdf_path in pdf_files:
            print("Get file hash")
            file_hash = self.get_file_hash(pdf_path)
            if pdf_path not in processed_meta or processed_meta[pdf_path] != file_hash:
                print(f" Mới hoặc thay đổi: {pdf_path}")
                new_files.append((pdf_path, file_hash))
            else:
                print(f" Bỏ qua (đã có): {pdf_path}")

        if not new_files:
            print("Không có file mới. Bỏ qua ingestion.")
            return None

        all_summarised_chunks = []

        for pdf_path, file_hash in new_files:
            # elements = partition_document(pdf_path)
            # chunks = create_chunks_by_title(elements)
            # all_summarised_chunks.extend(summarised_chunks)
            # summarised_chunks = summarise_chunks(chunks)
            elements = self.document_processing.partition_document(pdf_path)
            chunks = self.document_processing.create_chunks_by_title(elements)
            summarised_chunks = self.document_processing.summarise_chunks(chunks)
            processed_meta[pdf_path] = file_hash
            all_summarised_chunks.extend(summarised_chunks)

        
        # Save to DuckDB
        self.chunk_store.save_chunks(all_summarised_chunks)
        
        # Save embeddings
        # db = create_vector_store(all_summarised_chunks, persist_directory="db_agriculture2/chroma_db")
        db = self.embedding_execution.create_vector_store(all_summarised_chunks, persist_directory=self.EMBEDDING_VECTOR_DB_DIR)

        # Lưu metadata mới
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(processed_meta, f, ensure_ascii=False, indent=2)

        print("Incremental ingestion completed successfully!")
        return db
