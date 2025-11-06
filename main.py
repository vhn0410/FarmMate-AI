import glob
import json
import os
from typing import List
from src.rag_pipeline import RAGPipeline

# from src.implements import Datastore, Indexer, Retriever, ResponseGenerator, Evaluator
from src.implements.document_processing import DocumentProcessing
from src.implements.embedding import DocumentEmbedding
from src.implements.retriever import Retriever
from src.implements.chunk_store_duck_db import DuckDBChunkStore
from dotenv import load_dotenv
load_dotenv()

def create_pipeline() -> RAGPipeline:
    """Create and return a new RAG Pipeline instance with all components."""
    document_processing = DocumentProcessing()
    embedding_execution = DocumentEmbedding()
    # retriever = Retriever()
    # return RAGPipeline(document_processing, embedding_execution, retriever)
    return RAGPipeline(document_processing, embedding_execution)

def main():
    persist_directory = "db_agriculture3/chroma_db"
    pdf_dir = "./docs"
    
    # if not os.path.exists(persist_directory):
    pipeline = create_pipeline()
    db = pipeline.run_complete_ingestion_pipeline(pdf_dir=pdf_dir)
    
    print("Run complete ingestion pipeline")
    print("Success embedding all documents! Create new conversation with chatbot by command: uvicorn app:main --reload")
        
    
    chunk_store = DuckDBChunkStore("chunks.duckdb")
    documents = chunk_store.get_all_documents()
    print(documents[:2])
if __name__ == "__main__":
    main()
