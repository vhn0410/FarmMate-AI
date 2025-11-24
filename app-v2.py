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
# ✅ FIXED SSE generator với proper state handling
# --------------------------
async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    """
    ✅ FIXED: Streaming với state reset tự động cho mỗi turn mới
    
    Best Practices:
    1. Thread ID persistence cho multi-turn conversation
    2. Stream chỉ content từ response_generation và small_talk_response
    3. Proper error handling
    """
    try:
        # ✅ Generate hoặc reuse checkpoint
        is_new_conversation = checkpoint_id is None
        
        if is_new_conversation:
            checkpoint = str(uuid4())
            config = {"configurable": {"thread_id": checkpoint}}
            print(f"[APP] 🆕 New conversation started: {checkpoint}")
            yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{checkpoint}\"}}\n\n"
        else:
            checkpoint = checkpoint_id
            config = {"configurable": {"thread_id": checkpoint}}
            print(f"[APP] 🔄 Continuing conversation: {checkpoint}")

        # ✅ CRITICAL: Initial state với conversation_stage = "new_turn"
        # Điều này đảm bảo supervisor sẽ reset state cho mỗi message mới
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "original_query": message,
            "iteration_count": 0,
            "conversation_stage": "new_turn"  # ✅ Flag quan trọng!
        }

        # Create streaming event generator
        try:
            print(f"[APP] Starting stream for message: '{message}'")
            
            events = graph.astream_events(
                initial_state,
                version="v2",
                config=config
            )

            response_started = False
            response_ended = False
            current_node = None
            
            async for event in events:
                if response_ended:
                    break
                
                event_type = event.get("event")
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                event_data = event.get("data", {})

                # Track current node for debugging
                if node_name and node_name != current_node:
                    current_node = node_name
                    print(f"[APP] 📍 Now in node: {node_name}")

                # ✅ 1️⃣ Stream từ response_generation node (normal queries)
                if event_type == "on_chat_model_stream" and node_name == "response_generation":
                    if not response_started:
                        print(f"[APP] 🔄 Starting response stream...")
                        response_started = True
                    
                    chunk = event_data.get("chunk")
                    if chunk:
                        chunk_content = serialise_ai_message_chunk(chunk)
                        if chunk_content:
                            safe = safe_json_escape(chunk_content)
                            yield f"data: {{\"type\": \"content\", \"content\": \"{safe}\"}}\n\n"

                # ✅ 2️⃣ Detect response completion từ response_generation
                elif event_type == "on_chat_model_end" and node_name == "response_generation":
                    if response_started:
                        print(f"[APP] ✅ Response stream completed")
                        yield f"data: {{\"type\": \"end\"}}\n\n"
                        response_ended = True

                # ✅ 3️⃣ Handle small_talk_response (không stream, trả về ngay)
                elif event_type == "on_chain_end" and node_name == "small_talk_response":
                    output = event_data.get("output", {})
                    small_talk_response = output.get("draft_response", "")
                    
                    if small_talk_response:
                        print(f"[APP] 💬 Small talk response: {small_talk_response[:50]}...")
                        
                        # Stream từng ký tự để giống như chat thật
                        for char in small_talk_response:
                            safe_char = safe_json_escape(char)
                            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_char}\"}}\n\n"
                            await asyncio.sleep(0.01)  # Delay nhỏ để smooth
                        
                        yield f"data: {{\"type\": \"end\"}}\n\n"
                        response_ended = True

                # ✅ 4️⃣ Tool execution logs (optional, chỉ debug)
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    print(f"[APP] 🔧 Tool started: {tool_name}")
                
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    print(f"[APP] ✅ Tool completed: {tool_name}")

            # ✅ Nếu không có response nào được stream (edge case)
            if not response_started and not response_ended:
                print(f"[APP] ⚠️ No response was streamed, sending fallback")
                fallback = "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."
                for char in fallback:
                    safe_char = safe_json_escape(char)
                    yield f"data: {{\"type\": \"content\", \"content\": \"{safe_char}\"}}\n\n"
                yield f"data: {{\"type\": \"end\"}}\n\n"

        except asyncio.CancelledError:
            print(f"[APP] ⚠️ Request cancelled by client")
            yield f"data: {{\"type\": \"error\", \"message\": \"Request cancelled\"}}\n\n"
            return
        
        except Exception as e:
            print(f"[APP] ❌ Stream error: {e}")
            error_msg = safe_json_escape(str(e))
            yield f"data: {{\"type\": \"error\", \"message\": \"{error_msg}\"}}\n\n"

    except Exception as e:
        print(f"[APP] ❌ Fatal error: {e}")
        error_msg = safe_json_escape(str(e))
        yield f"data: {{\"type\": \"error\", \"message\": \"{error_msg}\"}}\n\n"


@app.get("/chat_stream")
async def chat_stream_get(message: str, checkpoint_id: Optional[str] = Query(None)):
    """
    GET endpoint for chat streaming
    
    Usage:
    - New conversation: /chat_stream?message=xin+chào
    - Continue conversation: /chat_stream?message=cảm+ơn&checkpoint_id=abc-123
    """
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
    """
    POST endpoint for chat streaming
    
    Body:
    {
        "message": "xin chào",
        "checkpoint_id": "abc-123"  // optional
    }
    """
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
    """Health check endpoint"""
    return {
        "status": "ok",
        "rag_enabled": True,
        "vector_db_loaded": vector_db is not None,
        "bm25_docs_count": len(documents),
        "graph_nodes": list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
    }

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Agricultural AI Agent API",
        "version": "2.0.0",
        "endpoints": {
            "GET /chat_stream": "Stream chat responses (SSE)",
            "POST /chat_stream": "Stream chat responses (SSE)",
            "GET /health": "Health check",
        },
        "features": [
            "Multi-turn conversation with memory",
            "Intent classification (small_talk, check_sensor, ask_knowledge, consultation)",
            "Tool orchestration (SensorThings API + Knowledge Base RAG)",
            "Quality assurance with self-improvement loop",
            "Streaming responses"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Agricultural AI Agent Server Starting...")
    print("=" * 60)
    print(f"📊 Vector DB: {'✅ Loaded' if vector_db else '❌ Not found'}")
    print(f"📚 BM25 Documents: {len(documents)}")
    print(f"🤖 LLM Model: gpt-4o")
    print(f"🧠 Router Model: gpt-4o-mini")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)