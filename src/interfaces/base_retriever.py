from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseRetriever(ABC):
    @abstractmethod
    def vector_search(self, query: str, top_k: int = 5):
        pass
    @abstractmethod
    def keyword_search(self, documents: list[Document], query: str, top_k: int = 5):
        pass
    @abstractmethod
    def hybrid_search(self, query):
        pass