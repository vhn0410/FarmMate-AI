from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List, Annotated, TypedDict, Any
from langgraph.graph import add_messages
from langchain_core.documents import Document
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_chroma import Chroma


class BaseEmbedding(ABC):
    @abstractmethod
    def create_vector_store(documents, persist_directory="dbv1/chroma_db") -> Chroma:
        pass
    @abstractmethod
    def load_vector_store(persist_directory="dbv1/chroma_db") -> Chroma: 
        pass