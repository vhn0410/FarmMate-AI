from pydantic import BaseModel
from abc import ABC, abstractmethod
from typing import List, Annotated, TypedDict, Any
from langgraph.graph import add_messages
from langchain_core.documents import Document
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

class DataItem(TypedDict):
    documents: Annotated[Document, add_messages]

class ContentData(TypedDict):
    text: str
    tables: List[Any]      
    images: List[Any]      
    types: List[str]

class BaseDocument(ABC):
    # Step 1: Partitioning PDF into atomic elements
    @abstractmethod
    def partition_document(file_path: str) -> list[Element]:
        pass
    # Step 2: Chunking by title
    @abstractmethod
    def create_chunks_by_title(elements) -> list[Element]:
        pass
    # Step 3: Summarising chunks and converting it into LangChain documents
    @abstractmethod
    def separate_content_types(chunk) -> ContentData:
        pass
    @abstractmethod
    def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
        pass
    def summarise_chunks(chunks) -> DataItem:
        pass
