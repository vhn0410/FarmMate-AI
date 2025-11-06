from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def load_vector_store(persist_directory="dbv1/chroma_db"):
    print("📂 Loading existing ChromaDB vector store...")
    
    # model_name = "BAAI/bge-small-en-v1.5"
    model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"   # or 4B, or 8B

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    print(f"✅ Loaded ChromaDB from {persist_directory}")
    return vectorstore