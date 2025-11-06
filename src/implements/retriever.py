from src.interfaces.base_retriever import BaseRetriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
load_dotenv()

class Retriever(BaseRetriever):

    def __init__(self, vector_retriever: Chroma, bm25_retriever):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def vector_search(self, query: str, top_k: int = 5):
        """Retrieve documents from vector Chroma database"""
        return self.vector_retriever.as_retriever(search_kwargs={"k": top_k})
    
    def keyword_search(self, documents: list[Document], query: str, top_k: int = 5):
        """Retrieve documents by keyword using bm25 algorithm"""
        self.bm25_retriever.k = top_k
        return self.bm25_retriever.invoke(query)
    
    def hybrid_search(self, query):
        retrievers = []
        weights = []
        if self.vector_retriever:
            retrievers.append(self.vector_retriever)
            weights.append(0.7)
        retrievers.append(self.bm25_retriever)
        weights.append(0.3)

        hybrid_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)

        return hybrid_retriever.invoke(query)
