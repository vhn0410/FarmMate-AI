import os
import json
import asyncio
from typing import TypedDict, Annotated, Literal, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymongo import AsyncMongoClient

# --- LangChain / LangGraph Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# --- Imports Tools của bạn ---
from src.tools.knowledge_base_tool import KnowledgeBaseService, build_kb_tool
from src.tools.sensorthings_tool import sensorthings_search

load_dotenv()

# ==========================================
# 1. DATA MODELS & STATE
# ==========================================

class UserProfile(BaseModel):
    name: Optional[str] = None
    crop_type: Optional[str] = None
    location: Optional[str] = None
    preferences: List[str] = Field(default_factory=list)

class QueryIntent(BaseModel):
    intent: Literal["small_talk", "ask_knowledge", "check_sensor"]
    enhanced_query: str
    data_needs: List[Literal["sensor_data", "knowledge_base"]] = Field(default_factory=list)

class SupervisorDecision(BaseModel):
    """Quyết định điều hướng của Supervisor"""
    next_node: Literal["memory_worker", "research_worker", "chat_worker", "FINISH"]
    reasoning: str

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    user_profile: dict
    original_query: str
    last_node: str         # Để Supervisor biết ai vừa báo cáo
    iteration_count: int
    final_response: str    # Câu trả lời cuối cùng để trả về

# ==========================================
# 2. SERVICES & WORKFLOW CONTAINER
# ==========================================

class AsyncMongoMemoryService:
    def __init__(self, uri, db_name):
        self.client = AsyncMongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db["user_profiles"]

    async def get_profile(self, user_id: str) -> dict:
        doc = await self.collection.find_one({"user_id": user_id})
        return UserProfile(**doc.get("profile", {})).model_dump() if doc else UserProfile().model_dump()

    async def update_profile(self, user_id: str, updates: dict):
        if not updates: return
        await self.collection.update_one(
            {"user_id": user_id},
            {"$set": {f"profile.{k}": v for k, v in updates.items() if v is not None}},
            upsert=True
        )

class Workflow:
    def __init__(self, retriever, llm, llm_router, mongo_uri, db_name):
        self.llm = llm            # GPT-4o
        self.llm_router = llm_router # GPT-4o-mini
        self.memory_service = AsyncMongoMemoryService(mongo_uri, db_name)
        
        # Tools
        self.kb_service = KnowledgeBaseService(retriever)
        self.kb_tool = build_kb_tool(self.kb_service)

# ==========================================
# 3. NODES (WORKERS & SUPERVISOR)
# ==========================================

# --- A. SUPERVISOR NODE (Bộ não) ---
async def supervisor_node(state: AgentState, config: RunnableConfig):
    wf: Workflow = config["configurable"]["workflow"]
    
    last_node = state.get("last_node")
    profile = state.get("user_profile")

    # 1. HEURISTICS (Luật cứng để tối ưu tốc độ)
    # Nếu chưa có profile -> Bắt buộc gọi Memory Worker
    if not profile or last_node == "start":
        return {"next_node": "memory_worker"}
    
    # Nếu Research hoặc Chat vừa làm xong -> FINISH (Kết thúc lượt)
    if last_node in ["research_worker", "chat_worker"]:
        return {"next_node": "end"}

    # 2. AI ROUTING (Suy luận)
    # Supervisor đọc lịch sử chat để chọn Worker
    system_prompt = """
    Bạn là Supervisor điều phối AI Nông nghiệp. Chọn worker phù hợp:
    - 'research_worker': Cho các câu hỏi về kỹ thuật, bệnh hại, giá cả, thời tiết (cần dùng Tool).
    - 'chat_worker': Cho các câu chào hỏi, cảm ơn, xã giao thông thường.
    """
    
    router = wf.llm_router.with_structured_output(SupervisorDecision)
    decision = await router.ainvoke([
        SystemMessage(content=system_prompt),
        *state["messages"][-5:] # Chỉ lấy 5 tin nhắn gần nhất context
    ])
    
    print(f"👮 Supervisor Decision: {decision.next_node} ({decision.reasoning})")
    
    if decision.next_node == "FINISH":
        return {"next_node": "end"}
        
    return {"next_node": decision.next_node}


# --- B. MEMORY WORKER ---
async def memory_worker(state: AgentState, config: RunnableConfig):
    wf: Workflow = config["configurable"]["workflow"]
    print("running memory_worker")
    user_id = state["user_id"]
    profile = await wf.memory_service.get_profile(user_id)
    
    return {
        "user_profile": profile,
        "last_node": "memory_worker",
        # Message ẩn để báo cho Supervisor biết đã xong
        "messages": [AIMessage(content=f"System: Loaded profile for {profile.get('name')}. Ready for request.")] 
    }


# --- C. RESEARCH WORKER (Gộp Intent -> Tool -> Gen -> QA) ---
async def research_worker(state: AgentState, config: RunnableConfig):
    """
    Worker này "thầu" toàn bộ quy trình trả lời câu hỏi chuyên môn.
    Việc gộp node giúp Supervisor không phải điều phối từng bước nhỏ.
    """
    wf: Workflow = config["configurable"]["workflow"]
    print("running research_worker")
    
    query = state["messages"][-1].content
    profile = state["user_profile"]
    
    # 1. Analyze Intent
    analyzer = wf.llm_router.with_structured_output(QueryIntent)
    intent = await analyzer.ainvoke(f"Context: {profile}\nQuery: {query}")
    
    # 2. Execute Tools (Parallel)
    tool_tasks = []
    if "knowledge_base" in intent.data_needs:
        tool_tasks.append(wf.kb_tool.ainvoke(intent.enhanced_query))
    if "sensor_data" in intent.data_needs:
        tool_tasks.append(sensorthings_search.ainvoke(intent.enhanced_query))
        
    raw_results = await asyncio.gather(*tool_tasks)
    tool_context = [str(r) for r in raw_results]
    
    # 3. Generate Answer
    prompt = f"""
    Answer user query: {state['original_query']}
    User Profile: {profile}
    Information retrieved: {tool_context}
    Be helpful, accurate and act as an agricultural expert.
    """
    response = await wf.llm.ainvoke(prompt)
    
    # 4. Update Memory (Side-effect)
    # (Có thể tách ra worker riêng nếu muốn logic phức tạp hơn)
    extractor = wf.llm_router.with_structured_output(UserProfile)
    updates = await extractor.ainvoke(f"Extract info from: {query}. Current: {profile}")
    await wf.memory_service.update_profile(state["user_id"], updates.model_dump(exclude_unset=True))

    return {
        "messages": [response], # Thêm câu trả lời vào lịch sử
        "final_response": response.content,
        "last_node": "research_worker"
    }


# --- D. CHAT WORKER (Small Talk) ---
async def chat_worker(state: AgentState, config: RunnableConfig):
    wf: Workflow = config["configurable"]["workflow"]
    print("running chat_worker")
    
    response = await wf.llm_router.ainvoke(
        [
            SystemMessage(content=f"Bạn là bạn đồng hành nhà nông. Tên user: {state['user_profile'].get('name')}."),
            HumanMessage(content=state["messages"][-1].content)
        ]
    )
    
    return {
        "messages": [response],
        "final_response": response.content,
        "last_node": "chat_worker"
    }

# ==========================================
# 4. GRAPH CONSTRUCTION
# ==========================================

def build_graph(mongo_uri: str):
    graph = StateGraph(AgentState)
    
    # Add Nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("memory_worker", memory_worker)
    graph.add_node("research_worker", research_worker)
    graph.add_node("chat_worker", chat_worker)
    
    # Set Entry
    graph.set_entry_point("supervisor")
    
    # Conditional Edges (Supervisor quyết định đi đâu)
    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next_node"],
        {
            "memory_worker": "memory_worker",
            "research_worker": "research_worker",
            "chat_worker": "chat_worker",
            "end": END
        }
    )
    
    # Normal Edges (Làm xong việc phải quay về báo cáo Supervisor)
    graph.add_edge("memory_worker", "supervisor")
    graph.add_edge("research_worker", "supervisor")
    graph.add_edge("chat_worker", "supervisor")
    
    # Checkpointer
    # mongo_client = AsyncMongoClient(mongo_uri)
    # checkpointer = AsyncMongoDBSaver(mongo_client)
    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)