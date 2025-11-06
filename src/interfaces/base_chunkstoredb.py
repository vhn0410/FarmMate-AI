from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List, Annotated, TypedDict, Any
from langgraph.graph import add_messages
from langchain_core.documents import Document
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_chroma import Chroma


class BaseDuckChunkStore(ABC):
    @abstractmethod
    def init_db(self):
        pass
    @abstractmethod
    def save_chunks(self, chunks):
        pass
    @abstractmethod
    def get_all_documents(self):
        pass