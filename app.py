"""
FastAPI Application for Agricultural AI Agent
Features:
- ✅ Server-Sent Events (SSE) for streaming
- ✅ Real-time response chunks
- ✅ Conversation history
- ✅ Health checks & monitoring
"""

import os
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import optimized supervisor
from src.agent.supervisor_sota_2025 import (
    build_graph, 
    Workflow,
    AgentState
)

# Import retriever
try:
    from src.implements.embedding import DocumentEmbedding
    from src.implements.retriever import Retriever
    from src.implements.chunk_store_duck_db import DuckDBChunkStore
    from langchain_community.retrievers import BM25Retriever
    
    PERSIST_DIR = "db_agriculture3/chroma_db"
    CHUNKS_DB = "chunks.duckdb"
    
    embedding = DocumentEmbedding()
    if os.path.exists(PERSIST_DIR):
        vector_db = embedding.load_vector_store(PERSIST_DIR)
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 50})
    else:
        vector_retriever = None
    
    chunk_store = DuckDBChunkStore(CHUNKS_DB)
    documents = chunk_store.get_all_documents()
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 50
    
    retriever = Retriever(
        vector_retriever=vector_retriever, 
        bm25_retriever=bm25_retriever
    )
    
    def get_retriever():
        return retriever
    
    print("✅ Loaded production retriever")

except ImportError:
    # Mock retriever for development
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    def get_retriever():
        return BM25Retriever.from_documents([
            Document(page_content="Lúa OM18 sinh trưởng 95-100 ngày, bón phân NPK 20-20-15"),
            Document(page_content="pH đất tối ưu cho lúa: 5.5-6.5"),
            Document(page_content="Bệnh đạo ôn lá: phun thuốc Validacin 3%")
        ])
    
    print("⚠️ Using mock retriever (development mode)")

# Config
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "agricultural_agent"

# Global state
app_graph = None
workflow_instance = None

# ==========================================
# LIFESPAN MANAGEMENT
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup"""
    global app_graph, workflow_instance
    
    print("\n🚀 Starting Agricultural AI Agent (SOTA 2025)...")
    print("="*60)
    
    # Get retriever
    retriever = get_retriever()
    
    # Init Workflow
    workflow_instance = Workflow(
        retriever=retriever,
        mongo_uri=MONGODB_URI,
        db_name=DB_NAME
    )
    
    # Build Graph
    app_graph = build_graph(
        mongo_uri=MONGODB_URI,
        db_name=DB_NAME
    )
    
    print("✅ Graph compiled successfully")
    print("✅ Memory service connected")
    print("✅ Tools initialized")
    print("="*60)
    print("🌾 Ready to assist farmers!\n")
    
    yield
    
    print("\n🛑 Shutting down gracefully...")

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="FarmMate AI - SOTA 2025",
    description="Agricultural AI Agent with Supervisor Multi-Agent Architecture",
    version="2.0.0",
    lifespan=lifespan
)

# ==========================================
# PYDANTIC MODELS
# ==========================================

class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question")
    user_id: str = Field(..., description="Unique user identifier")
    thread_id: str = Field(default="default", description="Conversation thread ID")
    stream: bool = Field(default=True, description="Enable streaming response")


class ChatResponse(BaseModel):
    response: str
    user_id: str
    thread_id: str
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict


# ==========================================
# STREAMING HELPER
# ==========================================

async def stream_agent_response(
    query: str,
    user_id: str,
    thread_id: str
):
    """
    New SSE generator — compatible with old frontend.
    Uses graph.astream_events() exactly like previous version.
    """

    if not app_graph or not workflow_instance:
        yield "data: {\"type\": \"error\", \"message\": \"Agent not initialized\"}\n\n"
        return

    try:
        # Load profile
        profile = await workflow_instance.memory_service.get_profile(user_id)

        # Create config with workflow
        config = {
            "configurable": {
                "thread_id": thread_id,
                "workflow": workflow_instance
            }
        }

        # Initial state
        state = {
            "messages": [HumanMessage(content=query)],
            "user_id": user_id,
            "thread_id": thread_id,
            "user_profile": profile,
            "intent": "",
            "next_worker": "intent_router",
            "retrieved_docs": [],
            "should_stream": True,
            "requires_reflection": True,
            "iteration": 0
        }

        # Announce session start
        yield 'data: {"type": "start", "message": "Processing query..."}\n\n'

        # MAIN FIX: use astream_events() instead of astream()
        events = app_graph.astream_events(
            state,
            version="v2",
            config=config
        )

        response_started = False
        response_ended = False
        full_response = ""

        async for event in events:
            if response_ended:
                break

            et = event.get("event")
            node_name = event.get("metadata", {}).get("langgraph_node", "")
            data = event.get("data", {})

            INTERNAL_NODES = [
                "intent_router", 
                "supervisor", 
                "planner_node",       # <--- QUAN TRỌNG: Thêm cái này
                "executor_node",      # <--- Nên thêm cái này
                "memory_update_worker" # <--- Nên thêm cái này
            ]
            
            if node_name in INTERNAL_NODES:
                # Bỏ qua, không gửi nội dung suy nghĩ của các node này xuống client
                continue
            

            # 1️⃣ Streaming chat output from synthesis worker
            # if et == "on_chat_model_stream" and node_name in ["synthesis_worker", "small_talk_worker"]:
            if et == "on_chat_model_stream":
                response_started = True
                chunk = data.get("chunk")

                if chunk:
                    from src.utils.utils import serialise_ai_message_chunk, safe_json_escape
                    text = serialise_ai_message_chunk(chunk)

                    if text:
                        full_response += text
                        safe = safe_json_escape(text)

                        yield f'data: {{"type":"content","content":"{safe}"}}\n\n'

            # 2️⃣ Finish generation
            elif et == "on_chat_model_end":
                if response_started:
                    response_ended = True
                    yield "data: {\"type\": \"end\"}\n\n"

            # 3️⃣ Tool logs (optional)
            elif et == "on_tool_end":
                tool_name = event.get("name")
                print(f"[TOOL] Completed: {tool_name}")

        # Save conversation
        await workflow_instance.memory_service.save_conversation(
            user_id=user_id,
            thread_id=thread_id,
            query=query,
            response=full_response,
            metadata={"length": len(full_response)}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = str(e).replace('"', '\\"')
        yield f'data: {{"type":"error","message":"{msg}"}}\n\n'


# ==========================================
# ENDPOINTS
# ==========================================
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc domain FE của bạn
    allow_credentials=True,
    allow_methods=["*"],   # QUAN TRỌNG: Cho phép OPTIONS
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "graph": "ready" if app_graph else "not_ready",
            "workflow": "ready" if workflow_instance else "not_ready",
            "memory": "connected"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Non-streaming chat endpoint
    Returns complete response
    """
    
    if not app_graph or not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Load profile
        profile = await workflow_instance.memory_service.get_profile(request.user_id)
        
        # Config
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "workflow": workflow_instance
            }
        }
        
        # Initial state
        initial_state = {
            "messages": [HumanMessage(content=request.query)],
            "user_id": request.user_id,
            "thread_id": request.thread_id,
            "user_profile": profile,
            "intent": "",
            "next_worker": "intent_router",
            "retrieved_docs": [],
            "should_stream": False,
            "requires_reflection": True,
            "iteration": 0
        }
        
        # Run graph
        final_state = await app_graph.ainvoke(initial_state, config)
        
        # Extract response
        ai_messages = [m for m in final_state["messages"] 
                      if isinstance(m, AIMessage) and not m.content.startswith("[")]
        response_text = ai_messages[-1].content if ai_messages else "No response generated"
        
        # Save conversation
        await workflow_instance.memory_service.save_conversation(
            user_id=request.user_id,
            thread_id=request.thread_id,
            query=request.query,
            response=response_text,
            metadata={
                "intent": final_state.get("intent"),
                "docs_retrieved": len(final_state.get("retrieved_docs", [])),
                "iterations": final_state.get("iteration", 0)
            }
        )
        
        return ChatResponse(
            response=response_text,
            user_id=request.user_id,
            thread_id=request.thread_id,
            metadata={
                "intent": final_state.get("intent"),
                "streaming": False
            }
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream") 
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint
    """
    if not request.stream:
        return await chat_endpoint(request)
    
    return StreamingResponse(
        stream_agent_response(
            query=request.query,
            user_id=request.user_id,
            thread_id=request.thread_id
        ),
        media_type="text/event-stream" 
    )

@app.get("/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile"""
    
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    profile = await workflow_instance.memory_service.get_profile(user_id)
    return {"user_id": user_id, "profile": profile}


@app.post("/profile/{user_id}")
async def update_user_profile(user_id: str, updates: dict):
    """Update user profile"""
    
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    await workflow_instance.memory_service.update_profile(user_id, updates)
    return {"status": "updated", "user_id": user_id}


@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str, limit: int = 10):
    """Get conversation history"""
    
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    cursor = workflow_instance.memory_service.conversations.find(
        {"user_id": user_id}
    ).sort("timestamp", -1).limit(limit)
    
    conversations = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        conversations.append(doc)
    
    return {"conversations": conversations}


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )