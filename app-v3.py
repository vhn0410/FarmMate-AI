import os
import sys
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# --- 1. SETUP PATH (Để import được module trong src) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)  # Thêm thư mục gốc vào path

# --- 2. IMPORTS TỪ MODULE SOTA CỦA BẠN ---
# Lưu ý: Đảm bảo tên file supervisor đúng như bạn đã lưu
from src.agent.supervisor_multiagent_SOTA import build_graph, Workflow
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Import Retriever của bạn (Nếu chưa có thì dùng Mock ở dưới)
try:
    # Giả sử bạn có file này để lấy Vector Store
    # from src.tools.vector_store import get_retriever
    from src.implements.embedding import DocumentEmbedding
    from src.implements.retriever import Retriever
    from src.implements.chunk_store_duck_db import DuckDBChunkStore
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever

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

    def get_retriever():
        return retriever
    
    print("✅ Đã import được Retriever thực tế.")
except ImportError:
    # --- MOCK RETRIEVER (Dùng tạm để App chạy được ngay) ---
    print("⚠️ Không tìm thấy module vector_store. Sử dụng Mock Retriever.")
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    def get_retriever():
        return BM25Retriever.from_documents([
            Document(page_content="Kỹ thuật trồng lúa OM18: Cần bón phân cân đối đạm, lân, kali. Thời gian sinh trưởng 95-100 ngày."),
            Document(page_content="Bệnh đạo ôn lá: Xuất hiện vết chấm kim, phát triển thành hình thoi. Phòng trị bằng thuốc đặc hiệu."),
            Document(page_content="Giá lúa hôm nay tại Cần Thơ: Dao động từ 7.800 - 8.200 đ/kg.")
        ])

# --- 3. CONFIGURATION ---
load_dotenv()
app_graph = None # Global variable lưu trữ graph đã compile

# --- 4. API MODELS (Pydantic) ---
class ChatRequest(BaseModel):
    query: str = Field(..., description="Câu hỏi của người dùng")
    user_id: str = Field(..., description="ID người dùng (để lưu Long-term memory)")
    thread_id: str = Field(default="default_thread", description="ID phiên chat (Short-term memory)")

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    user_id: str
    metadata: Optional[Dict[str, Any]] = None

# --- 5. LIFECYCLE (Khởi tạo 1 lần khi bật App) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Hàm này chạy 1 lần khi server khởi động.
    Dùng để khởi tạo Graph, DB connection để tiết kiệm tài nguyên.
    """
    global app_graph
    print("🚀 Đang khởi động FarmMate AI Agent...")
    
    try:
        # A. Khởi tạo LLM
        llm_smart = ChatOpenAI(model="gpt-4o", temperature=0)
        llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # B. Khởi tạo Retriever (Kiến thức nông nghiệp)
        retriever = get_retriever()
        
        # C. Khởi tạo Workflow (Dependency Injection)
        workflow_instance = Workflow(
            retriever=retriever,
            llm=llm_smart,
            llm_router=llm_fast
        )
        
        # D. Build Graph
        # Lưu ý: build_graph trong file SOTA đã bao gồm checkpointer MongoDB
        app_graph = build_graph(workflow=workflow_instance)
        
        print("✅ Agent Graph đã được khởi tạo thành công!")
        yield
        
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Agent: {e}")
        raise e
    finally:
        print("🛑 Shutting down FarmMate AI Agent...")

# --- 6. APP DEFINITION ---
app = FastAPI(
    title="FarmMate AI API",
    description="API cho AI Agent Nông nghiệp với Long-term Memory",
    version="2.0.0",
    lifespan=lifespan
)

# --- 7. ROUTES ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "agent_status": "ready" if app_graph else "not_initialized"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint chính để chat với Agent.
    """
    if not app_graph:
        raise HTTPException(status_code=500, detail="Agent chưa được khởi tạo.")
    
    # Cấu hình config cho LangGraph (Thread ID)
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Trạng thái đầu vào
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "original_query": request.query,
        "user_id": request.user_id,
        "iteration_count": 0,
        "last_node": "start",
        "memory_updated": False
    }
    
    try:
        # Gọi Agent (Dùng ainvoke để lấy kết quả cuối cùng)
        # Nếu muốn streaming, dùng app_graph.astream
        final_state = await app_graph.ainvoke(initial_state, config)
        
        # Trích xuất câu trả lời cuối cùng từ AI
        messages = final_state.get("messages", [])
        last_message = messages[-1]
        response_content = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
        
        # Lấy thêm metadata (nếu cần debug)
        metadata = {
            "intent": final_state.get("intent_data", {}).dict() if hasattr(final_state.get("intent_data"), "dict") else {},
            "qa_score": final_state.get("qa_feedback", {}).score if hasattr(final_state.get("qa_feedback"), "score") else None
        }
        
        return ChatResponse(
            response=response_content,
            thread_id=request.thread_id,
            user_id=request.user_id,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 8. RUN SERVER ---
if __name__ == "__main__":
    # Chạy server uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)