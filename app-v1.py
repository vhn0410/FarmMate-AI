from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
import os
from dotenv import load_dotenv
from typing import Optional, List
import json
import asyncio
from src.implements.embedding import DocumentEmbedding
from src.implements.retriever import Retriever
from src.implements.chunk_store_duck_db import DuckDBChunkStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.utils.utils import safe_json_escape, serialise_ai_message_chunk
from src.agent.supervisor_multiagent import Workflow, build_graph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# --------------------------
# Vector store & retrievers
# --------------------------
PERSIST_DIR = "db_agriculture3/chroma_db"
CHUNKS_DB = "chunks.duckdb"

embedding = DocumentEmbedding()

if os.path.exists(PERSIST_DIR):
    vector_db = embedding.load_vector_store(PERSIST_DIR)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 50})
else:
    vector_db = None
    vector_retriever = None

chunk_store = DuckDBChunkStore(CHUNKS_DB)
documents: List[Document] = chunk_store.get_all_documents()
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 50

retriever = Retriever(vector_retriever=vector_retriever, bm25_retriever=bm25_retriever)

# Khởi tạo LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------------
# Workflow & Graph
# --------------------------
workflow_instance = Workflow(retriever=retriever, llm=llm, llm_router=llm_router)
graph = build_graph(workflow_instance)

# --------------------------
# FastAPI setup
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

# --------------------------
# SSE generator (FIXED)
# --------------------------
async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    try:
        # Generate checkpoint
        is_new = checkpoint_id is None
        if is_new:
            checkpoint = str(uuid4())
            config = {"configurable": {"thread_id": checkpoint}}
            yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{checkpoint}\"}}\n\n"
        else:
            config = {"configurable": {"thread_id": checkpoint_id}}

        # Create streaming event generator
        try:
            print("[APP] Starting chat stream...")
            events = graph.astream_events(
                {"messages": [HumanMessage(content=message)]},
                version="v2",
                config=config
            )

            response_started = False
            response_ended = False
            print("[APP] Streaming events...")
            async for event in events:
                if response_ended:
                    break
                    
                et = event.get("event")
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                event_data = event.get("data", {})

                # --- 1️⃣ Stream chat model output từ response_generation ---
                if et == "on_chat_model_stream" and node_name == "response_generation":
                    response_started = True
                    chunk = event_data.get("chunk")
                    if chunk:
                        chunk_content = serialise_ai_message_chunk(chunk)
                        if chunk_content:
                            safe = safe_json_escape(chunk_content)
                            yield f"data: {{\"type\": \"content\", \"content\": \"{safe}\"}}\n\n"

                # --- 2️⃣ Detect tool calls (search_start) ---
                elif et == "on_chat_model_end" and node_name == "response_generation":
                    output = event_data.get("output")
                    
                    # Nếu AI đã stream response
                    if response_started:
                        yield f"data: {{\"type\": \"end\"}}\n\n"
                        response_ended = True

                # --- 3️⃣ Tool result (search_results) - CHỈ LOG, KHÔNG GỬI ---
                elif et == "on_tool_end":
                    tool_name = event.get("name")
                    print(f"[APP] Tool {tool_name} completed")
                    # KHÔNG yield search_results nữa vì không cần thiết

        except asyncio.CancelledError:
            yield f"data: {{\"type\": \"error\", \"message\": \"Request cancelled\"}}\n\n"
            return

    except Exception as e:
        error_msg = safe_json_escape(str(e))
        yield f"data: {{\"type\": \"error\", \"message\": \"{error_msg}\"}}\n\n"


@app.get("/chat_stream")
async def chat_stream_get(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        },
    )

@app.post("/chat_stream")
async def chat_stream_post(req: ChatRequest):
    return StreamingResponse(
        generate_chat_responses(req.message, req.checkpoint_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        },
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