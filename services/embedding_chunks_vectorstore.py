from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("🔮 Creating embeddings and storing in ChromaDB...")
        
    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    # model_name = "BAAI/bge-small-en-v1.5"   # or 4B, or 8B
    model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"   # or 4B, or 8B

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cuda"                    # chạy trên GPU
        },
        # encode_kwargs={
        #     "normalize_embeddings": True     # giúp Chroma hoạt động ổn định hơn
        # },
        show_progress=True  # Thêm progress bar cho embedding model
    )
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore

# Create the vector store
# db = create_vector_store(processed_chunks)