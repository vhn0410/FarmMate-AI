"""
Improved RAG Agent application (LangGraph + FastAPI) for agriculture.
- Architecture: Planner -> Router -> Tool Node -> Responder
- Fixes: append messages (preserve history), LLM-based router/planner, tool output validation & truncation,
  chunked sensorthings responses, retriever returns docs (not full prompt), and responder assembles final answer.
- Streaming SSE endpoint preserved.
- FIXED: No duplicate responses - only responder streams output

Note: adapt model classes (ChatOpenAI, ChatGoogleGenerativeAI) to your available SDKs.

"""

from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import itertools
import json
import os
import requests
from uuid import uuid4
from dotenv import load_dotenv

# RAG imports (assume these are available in your env)
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from src.implements.embedding import DocumentEmbedding
from src.implements.retriever import Retriever
from src.utils.utils import safe_json_escape, serialise_ai_message_chunk
from src.tools.sensorthings_tool import sensorthings_search
from src.tools.knowledge_base_tool import KnowledgeBaseService, build_kb_tool
from src.agent.workflow import graph
load_dotenv()

# --------------------------
# Configuration
# --------------------------
PERSIST_DIR = "db_agriculture3/chroma_db"
CHUNKS_DB = "chunks.duckdb"
# --------------------------
# Vector store & retrievers (initialize at startup)
# --------------------------
print("🔄 Initializing vector store and retrievers...")
embedding = DocumentEmbedding()

if os.path.exists(PERSIST_DIR):
    vector_db = embedding.load_vector_store(PERSIST_DIR)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 15})
    # vector_retriever = vector_db.as_retriever(
    #     search_type="mmr",
    #     search_kwargs={
    #         "k": 15,          # tổng số doc muốn xét
    #         "fetch_k": 50,    # lấy nhiều trước, rồi filter lại (optional)
    #         "lambda_mult": 0.5  # độ cân bằng giữa relevance và diversity
    #     }
    # )
else:
    print("⚠️ Vector DB not found. Vector retriever disabled.")
    vector_db = None
    vector_retriever = None
# --------------------------
# Preload chunk documents for BM25
# --------------------------
from src.implements.chunk_store_duck_db import DuckDBChunkStore
chunk_store = DuckDBChunkStore(CHUNKS_DB)
documents: List[Document] = chunk_store.get_all_documents()
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15
retriever = Retriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)


# --------------------------
# FastAPI & SSE
# --------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    checkpoint_id: Optional[str] = None


async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    """Generate streaming responses. Only responder node should stream user-facing content."""
    try:
        is_new = checkpoint_id is None
        if is_new:
            checkpoint = str(uuid4())
            config = {"configurable": {"thread_id": checkpoint}}
            yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{checkpoint}\"}}\n\n"
        else:
            config = {"configurable": {"thread_id": checkpoint_id}}

        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]}, 
            version="v2", 
            config=config
        )

        async for event in events:
            et = event["event"]
            
            if et == "on_chat_model_stream":
                try:
                    # Get the node name from metadata to filter which node is streaming
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    
                    # CRITICAL: Only stream output from responder node
                    # Skip streaming from planner to avoid duplicate responses
                    if node_name != "responder":
                        continue
                    
                    chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
                    if not chunk_content:
                        continue
                        
                    safe = safe_json_escape(chunk_content)
                    yield f"data: {{\"type\": \"content\", \"content\": \"{safe}\"}}\n\n"
                    
                except Exception as e:
                    # Silently skip problematic chunks
                    continue
                    
            elif et == "on_chat_model_end":
                try:
                    # Get node name
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    
                    # Only process tool calls from planner node
                    if node_name != "planner":
                        continue
                    
                    output = event["data"]["output"]
                    tool_calls = getattr(output, "tool_calls", [])
                    
                    # Notify about RAG search if it's happening
                    rag_calls = [c for c in tool_calls if c["name"] == "search_knowledge_base"]
                    if rag_calls:
                        q = rag_calls[0]["args"].get("query", "")
                        yield f"data: {{\"type\": \"rag_search_start\", \"query\": \"{safe_json_escape(str(q))}\"}}\n\n"
                        
                except Exception:
                    pass
                    
            elif et == "on_tool_end":
                try:
                    name = event.get("name")
                    output = event["data"]["output"]
                    
                    # Create a short preview of tool output
                    preview = str(output)
                    if len(preview) > 500:
                        preview = preview[:500] + "..."
                        
                    yield f"data: {{\"type\": \"tool_complete\", \"tool\": \"{name}\", \"preview\": \"{safe_json_escape(preview)}\"}}\n\n"
                    
                except Exception:
                    pass
                    
        yield f"data: {{\"type\": \"end\"}}\n\n"
        
    except Exception as e:
        yield f"data: {{\"type\": \"error\", \"message\": \"{safe_json_escape(str(e))}\"}}\n\n"



@app.get("/chat_stream")
async def chat_stream_get(message: str, checkpoint_id: Optional[str] = Query(None)):
    """GET endpoint - legacy support. Use POST for better reliability."""
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/chat_stream")
async def chat_stream_post(req: ChatRequest):
    """POST endpoint for production use."""
    return StreamingResponse(
        generate_chat_responses(req.message, req.checkpoint_id), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "rag_enabled": True,
        "vector_db_loaded": vector_db is not None,
        "bm25_docs_count": len(documents)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)