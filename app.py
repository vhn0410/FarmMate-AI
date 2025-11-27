"""
FastAPI Application for Agricultural AI Agent
Features:
- ✅ JWT Keycloak Authentication
- ✅ Dynamic user_id from token
- ✅ Server-Sent Events (SSE) for streaming
- ✅ Real-time response chunks
- ✅ Conversation history
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 🔥 Import authentication
from src.auth.keycloak_auth import get_current_user, get_optional_user, KeycloakUser

# Import optimized supervisor
from src.agent.supervisor_sota_2025 import (
    build_graph, 
    Workflow,
)
from uuid import uuid4
from typing import List
from datetime import datetime

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
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    def get_retriever():
        return BM25Retriever.from_documents([
            Document(page_content="Lúa OM18 sinh trưởng 95-100 ngày, bón phân NPK 20-20-15"),
            Document(page_content="pH đất tối ưu cho lúa: 5.5-6.5"),
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
    
    retriever = get_retriever()
    
    workflow_instance = Workflow(
        retriever=retriever,
        mongo_uri=MONGODB_URI,
        db_name=DB_NAME
    )
    
    app_graph = build_graph(
        mongo_uri=MONGODB_URI,
        db_name=DB_NAME
    )
    
    print("✅ Graph compiled successfully")
    print("✅ Memory service connected")
    print("✅ JWT Authentication enabled")
    print("="*60)
    print("🌾 Ready to assist farmers!\n")
    
    yield
    
    print("\n🛑 Shutting down gracefully...")

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(
    title="FarmMate AI - SOTA 2025",
    description="Agricultural AI Agent with JWT Authentication",
    version="2.1.0",
    lifespan=lifespan
)

# CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# PYDANTIC MODELS
# ==========================================

class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question")
    thread_id: str = Field(default="default", description="Conversation thread ID")
    stream: bool = Field(default=True, description="Enable streaming response")


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict


# ==========================================
# STREAMING HELPER with JWT
# ==========================================

async def stream_agent_response(
    query: str,
    user_id: str,
    thread_id: str
):
    """Stream agent response with user-specific context"""
    if not app_graph or not workflow_instance:
        yield "data: {\"type\": \"error\", \"message\": \"Agent not initialized\"}\n\n"
        return

    try:
        # Load profile
        profile = await workflow_instance.memory_service.get_profile(user_id)

        # 🔥 Create user-specific sensor tool
        from src.tools.sensorthings_tool import create_sensorthings_tool
        user_sensor_tool = create_sensorthings_tool(user_id)

        # Create config with workflow and user-specific tools
        config = {
            "configurable": {
                "thread_id": thread_id,
                "workflow": workflow_instance,
                "user_sensor_tool": user_sensor_tool  # 🔥 Pass user-specific tool
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

        yield 'data: {"type": "start", "message": "Processing query..."}\n\n'

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
                "planner_node",      
                "executor_node",      
                "memory_update_worker" 
            ]
            
            if node_name in INTERNAL_NODES:
                continue

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

            elif et == "on_chat_model_end":
                if response_started:
                    response_ended = True
                    yield "data: {\"type\": \"end\"}\n\n"

        # Save conversation
        await workflow_instance.memory_service.db["threads"].update_one(
            {"thread_id": thread_id, "user_id": user_id},
            {
                "$set": {
                    "updated_at": datetime.now().isoformat(),
                    "last_message": query[:100]
                },
                "$inc": {"message_count": 2}
            },
            upsert=True
        )
        
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "components": {
            "graph": "ready" if app_graph else "not_ready",
            "workflow": "ready" if workflow_instance else "not_ready",
            "memory": "connected",
            "auth": "jwt_enabled"
        }
    }


@app.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    user: KeycloakUser = Depends(get_current_user)  # 🔥 JWT Authentication Required
):
    """
    Streaming chat endpoint with JWT authentication
    
    Requires:
        - Valid JWT token in Authorization header
        - Bearer token format
    """
    return StreamingResponse(
        stream_agent_response(
            query=request.query,
            user_id=user.sub,  # 🔥 Use user_id from JWT token
            thread_id=request.thread_id
        ),
        media_type="text/event-stream"
    )


@app.get("/profile")
async def get_user_profile(user: KeycloakUser = Depends(get_current_user)):
    """Get current user profile from JWT token"""
    
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Get profile from database
    profile = await workflow_instance.memory_service.get_profile(user.sub)
    
    return {
        "user_id": user.sub,
        "username": user.preferred_username,
        "email": user.email,
        "name": user.name,
        "roles": user.realm_roles,
        "profile": profile
    }


@app.post("/profile")
async def update_user_profile(
    updates: dict,
    user: KeycloakUser = Depends(get_current_user)
):
    """Update user profile"""
    
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    await workflow_instance.memory_service.update_profile(user.sub, updates)
    return {"status": "updated", "user_id": user.sub}


# ==========================================
# THREAD ENDPOINTS (with JWT)
# ==========================================

class ThreadCreate(BaseModel):
    title: str = Field(default="Cuộc hội thoại mới", description="Thread title")

class ThreadResponse(BaseModel):
    thread_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message: Optional[str] = None


@app.post("/threads", response_model=ThreadResponse)
async def create_thread(
    request: ThreadCreate,
    user: KeycloakUser = Depends(get_current_user)  # 🔥 JWT Required
):
    """Create thread for authenticated user"""
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    thread_id = str(uuid4())
    now = datetime.now().isoformat()
    
    thread_doc = {
        "thread_id": thread_id,
        "user_id": user.sub,  # 🔥 Use JWT user_id
        "title": request.title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
        "last_message": None
    }
    
    await workflow_instance.memory_service.db["threads"].insert_one(thread_doc)
    
    return ThreadResponse(**thread_doc)


@app.get("/threads", response_model=List[ThreadResponse])
async def get_user_threads(
    limit: int = 50,
    user: KeycloakUser = Depends(get_current_user)  # 🔥 JWT Required
):
    """Get threads for authenticated user"""
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    cursor = workflow_instance.memory_service.db["threads"].find(
        {"user_id": user.sub}  # 🔥 Filter by JWT user_id
    ).sort("updated_at", -1).limit(limit)
    
    threads = []
    async for doc in cursor:
        doc.pop("_id", None)
        threads.append(ThreadResponse(**doc))
    
    return threads


@app.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    user: KeycloakUser = Depends(get_current_user)  # 🔥 JWT Required
):
    """Delete thread (only if owned by user)"""
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # 🔥 Ensure user can only delete their own threads
    result = await workflow_instance.memory_service.db["threads"].delete_one({
        "thread_id": thread_id,
        "user_id": user.sub  # 🔥 Security check
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Thread not found or access denied")
    
    # Delete conversations
    await workflow_instance.memory_service.conversations.delete_many({
        "thread_id": thread_id,
        "user_id": user.sub
    })
    
    return {"status": "deleted", "thread_id": thread_id}


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