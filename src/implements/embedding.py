from src.interfaces.base_embedding import BaseEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class DocumentEmbedding(BaseEmbedding):

    EMBEDDING_MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"

    def create_vector_store(self, documents, persist_directory="dbv1/chroma_db") -> Chroma:
        """Create and persist ChromaDB vector store"""
        print("🔮 Creating embeddings and storing in ChromaDB...")
            
        # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        # model_name = "BAAI/bge-small-en-v1.5"   # or 4B, or 8B

        embedding_model = HuggingFaceEmbeddings(
            model_name = self.EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": "cuda"                     # Chạy trên GPU
            },
            encode_kwargs={
                "normalize_embeddings": True         # Giúp Chroma hoạt động ổn định hơn
            },
            show_progress=True                       # Thêm progress bar cho embedding model
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

    def load_vector_store(self, persist_directory="dbv1/chroma_db") -> Chroma:
        print("📂 Loading existing ChromaDB vector store...")


        embedding_model = HuggingFaceEmbeddings(
            model_name = self.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"}
        )

        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )

        print(f"✅ Loaded ChromaDB from {persist_directory}")
        return vectorstore