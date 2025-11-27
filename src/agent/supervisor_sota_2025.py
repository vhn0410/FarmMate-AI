"""
SOTA Supervisor with Sequential Tool Execution (2025)

🎯 KEY IMPROVEMENT: Plan → Execute Sensors → Execute KB → Synthesize
- Sensors data FIRST, then use that context for KB retrieval
- Multi-step reasoning with explicit planning
- Dynamic context engineering

Based on:
- Plan-and-Execute pattern (Wang et al.)
- LLMCompiler DAG execution
- LangGraph sequential workflows
"""

import os
import json
import asyncio
from typing import TypedDict, Annotated, Literal, List, Optional, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymongo import AsyncMongoClient

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, 
    SystemMessage, BaseMessage
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from src.tools.knowledge_base_tool import KnowledgeBaseService, build_kb_tool
from src.tools.sensorthings_tool import sensorthings_search

load_dotenv()

# ==========================================
# 1. ENHANCED STATE WITH EXECUTION PLAN
# ==========================================

class AgentState(TypedDict):
    """State with explicit execution plan"""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    thread_id: str
    
    # Intent & Routing
    intent: str
    next_worker: str
    
    # 🔥 NEW: Execution Plan
    execution_plan: List[str]  # ["sensor_tool", "kb_tool"]
    current_step: int
    
    # Tool Results (accumulated)
    sensor_data: Optional[Dict]  # Raw sensor readings
    kb_context: Optional[str]    # Knowledge base info
    
    retrieved_docs: List[Dict]
    user_profile: dict
    
    # Control
    should_stream: bool
    requires_reflection: bool
    iteration: int


# ==========================================
# 2. PYDANTIC MODELS
# ==========================================

class UserProfile(BaseModel):
    user_id: str
    name: Optional[str] = None
    farm_location: Optional[str] = None
    crop_types: List[str] = Field(default_factory=list)
    sensors: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class IntentResult(BaseModel):
    intent: Literal["small_talk", "knowledge", "sensor", "consultation"]
    confidence: float
    reasoning: str
    enhanced_query: str


class ExecutionPlan(BaseModel):
    steps: List[Literal["sensor_tool", "kb_tool"]]
    reasoning: str
    sensor_query: Optional[str] = None
    kb_query: Optional[str] = None


class ReflectionResult(BaseModel):
    score: int = Field(ge=0, le=100)
    feedback: str
    strengths: List[str]
    weaknesses: List[str]


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    crop_types: Optional[List[str]] = None
    farm_location: Optional[str] = None
    sensors: Optional[List[str]] = None


# ==========================================
# 3. MEMORY SERVICE
# ==========================================

class AsyncMemoryService:
    def __init__(self, uri: str, db_name: str):
        self.client = AsyncMongoClient(uri)
        self.db = self.client[db_name]
        self.profiles = self.db["user_profiles"]
        self.conversations = self.db["conversations"]
        self._cache = {}
    
    async def get_profile(self, user_id: str) -> dict:
        if user_id in self._cache:
            return self._cache[user_id]
        
        doc = await self.profiles.find_one({"user_id": user_id})
        if doc:
            profile = doc.get("profile", {})
        else:
            profile = UserProfile(user_id=user_id).model_dump()
        
        self._cache[user_id] = profile
        return profile
    
    async def update_profile(self, user_id: str, updates: dict):
        if not updates:
            return
        
        updates["last_updated"] = datetime.now().isoformat()
        await self.profiles.update_one(
            {"user_id": user_id},
            {"$set": {f"profile.{k}": v for k, v in updates.items()}},
            upsert=True
        )
        
        if user_id in self._cache:
            del self._cache[user_id]
    
    async def save_conversation(self, user_id: str, thread_id: str, 
                               query: str, response: str, metadata: dict):
        await self.conversations.insert_one({
            "user_id": user_id,
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata
        })


# ==========================================
# 4. INTENT ROUTER
# ==========================================

async def intent_router_node(state: AgentState, config: RunnableConfig):
    wf = config["configurable"]["workflow"]
    query = state["messages"][-1].content
    profile = state["user_profile"]
    
    # SOTA Prompt: Role Definition + Explicit Constraints + Few-Shot Examples (Vietnamese)
    system_prompt = f"""You are an Intent Classifier for a Vietnamese Agricultural AI.

USER CONTEXT:
- Name: {profile.get('name', 'Unknown')}
- Crops: {', '.join(profile.get('crop_types', ['Not specified']))}

CLASSIFICATION RULES:
1. **small_talk**: Social interactions, greetings, thanks.
   - Examples: "Chào bạn", "Cảm ơn nhé", "Bạn tên gì?"
2. **knowledge**: General farming theory/techniques. NO real-time data needed.
   - Examples: "Cách trồng lúa", "Bệnh đạo ôn là gì?", "Quy trình bón phân cho xoài"
3. **sensor**: User asks for SPECIFIC numbers/logs only.
   - Examples: "Nhiệt độ hiện tại bao nhiêu?", "Cho xem log độ ẩm hôm qua", "Đất có chua không?" (Needs pH value)
4. **consultation**: COMPLEX requests requiring Analysis, Diagnosis, or Recommendations based on CURRENT status.
   - Examples: "Cây của tôi bị vàng lá, phải làm sao?", "Kiểm tra xem dinh dưỡng đất có ổn cho cây sầu riêng không?", "Phân tích tình trạng vườn".

CRITICAL LOGIC (Chain of Thought):
- If user mentions "bệnh" (disease) or "kiểm tra" (check) -> likely 'consultation' because we need sensor data + KB diagnosis.
- If user asks purely about "lý thuyết" (theory) -> 'knowledge'.

Output JSON."""
    
    recent_messages = state["messages"][-5:]
    router = wf.llm_fast.with_structured_output(IntentResult)
    result = await router.ainvoke([
        SystemMessage(content=system_prompt),
    ] + recent_messages)
    
    print(f"🔍 Intent: {result.intent} ({result.confidence:.0%}) - {result.reasoning}")
    
    # Route logic giữ nguyên
    if result.intent == "small_talk":
        next_worker = "small_talk_worker"
    elif result.intent in ["sensor", "consultation"]:
        next_worker = "planner_node"
    else:
        next_worker = "retrieval_worker"
    
    return {
        "intent": result.intent,
        "next_worker": next_worker,
        "messages": [AIMessage(
            content=f"[Intent: {result.intent}]",
            additional_kwargs={"intent_meta": result.model_dump()}
        )]
    }


# ==========================================
# 5. 🔥 NEW: PLANNER NODE
# ==========================================

async def planner_node(state: AgentState, config: RunnableConfig):
    wf = config["configurable"]["workflow"]
    query = state["messages"][-1].content
    intent = state["intent"]
    
    # SOTA Prompt: Task Decomposition
    system_prompt = f"""You are an Execution Planner for an Agricultural Agent.
    
GOAL: Create a step-by-step plan to answer the user's Vietnamese query.

AVAILABLE TOOLS:
1. "sensor_tool": Get real-time data (Temp, Humidity, NPK, pH, EC).
2. "kb_tool": Search farming manuals, disease databases, pest control guidelines.

LOGIC (Chain of Thought):
- If Intent is 'consultation' (e.g., "Why is my plant yellow?"):
  1. I need to know the current environment status (Is it too hot? Soil too acid?) -> Call "sensor_tool".
  2. Then I need to match those conditions with disease symptoms in the database -> Call "kb_tool".
  -> Plan: ["sensor_tool", "kb_tool"]

- If Intent is 'sensor' (e.g., "Current pH?"):
  -> Plan: ["sensor_tool"]

- If Intent is 'knowledge' (e.g., "How to plant rice?"):
  -> Plan: ["kb_tool"]

INSTRUCTIONS:
- 'sensor_query': Translate user intent into specific metrics (e.g., "Get temperature, humidity, soil moisture").
- 'kb_query': Formulate a search query. If sensor data will be available, write a query that utilizes it (e.g., "Diagnosis for yellow leaves with high soil moisture").

Return JSON."""
    
    planner = wf.llm_fast.with_structured_output(ExecutionPlan)
    plan = await planner.ainvoke(system_prompt)
    
    print(f"\n📋 === EXECUTION PLAN ===")
    print(f"Steps: {' → '.join(plan.steps)}")
    print(f"Reasoning: {plan.reasoning}")
    if plan.sensor_query:
        print(f"Sensor Query: {plan.sensor_query}")
    if plan.kb_query:
        print(f"KB Query: {plan.kb_query}")
    
    return {
        "execution_plan": plan.steps,
        "current_step": 0,
        "next_worker": "executor_node",
        "messages": [AIMessage(
            content=f"[Plan: {' → '.join(plan.steps)}]",
            additional_kwargs={
                "plan": plan.model_dump()
            }
        )]
    }


# ==========================================
# 6. 🔥 NEW: EXECUTOR NODE (Sequential)
# ==========================================

async def executor_node(state: AgentState, config: RunnableConfig):
    """
    Executes tools sequentially according to plan
    Each step can use results from previous steps
    """
    wf = config["configurable"]["workflow"]
    plan = state["execution_plan"]
    current_step = state["current_step"]
    
    if current_step >= len(plan):
        # All steps executed
        return {
            "next_worker": "synthesis_worker"
        }
    
    tool_name = plan[current_step]
    print(f"\n🔧 === EXECUTING STEP {current_step + 1}/{len(plan)}: {tool_name} ===")
    
    # Get plan metadata from last AI message
    plan_msg = [m for m in state["messages"] if isinstance(m, AIMessage) 
                and m.additional_kwargs.get("plan")][-1]
    plan_data = plan_msg.additional_kwargs["plan"]
    
    # Execute appropriate tool
    if tool_name == "sensor_tool":
        sensor_query = plan_data.get("sensor_query") or state["messages"][-1].content
        
        
        # result = await sensorthings_search.ainvoke(sensor_query)
        
        user_sensor_tool = config["configurable"].get("user_sensor_tool")
        
        if user_sensor_tool:
            # Use user-specific tool (with dynamic user_id)
            result = await user_sensor_tool.ainvoke(sensor_query)
        else:
            # Fallback to default tool (for backward compatibility)
            from src.tools.sensorthings_tool import sensorthings_search
            result = await sensorthings_search.ainvoke(sensor_query)
            

        if result and isinstance(result, dict):
            sensor_summary = json.dumps(result, ensure_ascii=False, indent=2)
            # tool_msg = ToolMessage(
            #     content=f"[SENSOR DATA]\n{sensor_summary}",
            #     tool_call_id="sensor_tool"
            # )
            # 🔥 FIX: Dùng HumanMessage thay vì ToolMessage
            tool_msg = HumanMessage(
                content=f"📋 [SYSTEM - SENSOR DATA]:\n{sensor_summary}",
                name="sensor_tool" # name giúp LLM phân biệt nguồn
            )
            print(f"✅ Sensors: {len(result)} readings retrieved")
            
            return {
                "sensor_data": result,
                "messages": [tool_msg],
                "current_step": current_step + 1,
                "next_worker": "executor_node",  # Continue to next step
                "retrieved_docs": state.get("retrieved_docs", []) + [{
                    "source": "sensors",
                    "data": result
                }]
            }
        else:
            # return {
            #     "messages": [ToolMessage(
            #         content="[SENSOR DATA] No sensor data available",
            #         tool_call_id="sensor_tool"
            #     )],
            #     "current_step": current_step + 1,
            #     "next_worker": "executor_node"
            # }
            # 🔥 FIX: Dùng HumanMessage
            print("⚠️ Sensors: No data")
            return {
                "messages": [HumanMessage(content="📋 [SYSTEM - SENSOR]: No data available")],
                "current_step": current_step + 1,
                "next_worker": "executor_node"
            }
    
    elif tool_name == "kb_tool":
        # 🔥 CRITICAL: Use sensor context to enhance KB query
        kb_query = plan_data.get("kb_query") or state["messages"][-1].content
        
        # Enrich KB query with sensor results
        if state.get("sensor_data"):
            sensor_summary = json.dumps(state["sensor_data"], ensure_ascii=False)
            enhanced_query = f"""Based on these sensor readings:
{sensor_summary}

{kb_query}"""
            print(f"🔍 Enhanced KB Query with sensor context")
        else:
            enhanced_query = kb_query
        
        result = await wf.kb_tool.ainvoke(enhanced_query)
        
        if result and len(result.strip()) > 10:
            # tool_msg = ToolMessage(
            #     content=f"[KNOWLEDGE BASE]\n{result}",
            #     tool_call_id="kb_tool"
            # )

            # 🔥 FIX: Dùng HumanMessage thay vì ToolMessage
            tool_msg = HumanMessage(
                content=f"📚 [SYSTEM - KNOWLEDGE BASE]:\n{result}",
                name="kb_tool"
            )
            print(f"✅ KB: {len(result)} chars retrieved")
            
            return {
                "kb_context": result,
                "messages": [tool_msg],
                "current_step": current_step + 1,
                "next_worker": "executor_node",
                "retrieved_docs": state.get("retrieved_docs", []) + [{
                    "source": "knowledge_base",
                    "content": result[:500]
                }]
            }
        else:
            print("⚠️ KB: No results")
            # return {
            #     "messages": [ToolMessage(
            #         content="[KNOWLEDGE BASE] No relevant information found",
            #         tool_call_id="kb_tool"
            #     )],
            #     "current_step": current_step + 1,
            #     "next_worker": "executor_node"
            # }
            # 🔥 FIX: Dùng HumanMessage
            return {
                "messages": [HumanMessage(content="📚 [SYSTEM - KB]: No relevant info found")],
                "current_step": current_step + 1,
                "next_worker": "executor_node"
            }


# ==========================================
# 7. SMALL TALK WORKER
# ==========================================

async def small_talk_worker(state: AgentState, config: RunnableConfig):
    wf = config["configurable"]["workflow"]
    profile = state["user_profile"]
    name = profile.get("name", "bạn")
    
    system_prompt = f"""Bạn là trợ lý ảo nông nghiệp vui tính, thân thiện.
    
    Người dùng: {name}
    Nhiệm vụ: Trò chuyện xã giao, chào hỏi.
    Phong cách: Ngắn gọn, ấm áp, đậm chất miền Tây hoặc nông thôn Việt Nam.
    Ví dụ: "Chào bác Ba, hôm nay lúa má thế nào rồi ạ? 🌾", "Dạ con nghe nè, bác cần giúp gì hông?"
    
    HÃY TRẢ LỜI BẰNG TIẾNG VIỆT."""
    
    # ✅ FIX: Ghép System Message với toàn bộ lịch sử chat hiện có
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = await wf.llm_fast.ainvoke(messages)
    
    return {
        "messages": [response],
        "next_worker": "END"
    }


# ==========================================
# 8. LEGACY RETRIEVAL WORKER (for knowledge-only)
# ==========================================

async def retrieval_worker(state: AgentState, config: RunnableConfig):
    """Direct KB retrieval for pure knowledge queries"""
    wf = config["configurable"]["workflow"]
    query = state["messages"][-1].content
    
    print("\n🔧 === KNOWLEDGE RETRIEVAL ===")
    
    result = await wf.kb_tool.ainvoke(query)
    
    if result and len(result.strip()) > 10:
        # tool_msg = ToolMessage(
        #     content=f"[KNOWLEDGE BASE]\n{result}",
        #     tool_call_id="kb_tool"
        # )
        # 🔥 FIX: Dùng HumanMessage
        tool_msg = HumanMessage(
            content=f"📚 [SYSTEM - KNOWLEDGE BASE]:\n{result}",
            name="kb_tool"
        )
        print(f"✅ KB: {len(result)} chars")
        
        return {
            "messages": [tool_msg],
            "kb_context": result,
            "retrieved_docs": [{
                "source": "knowledge_base",
                "content": result[:500]
            }],
            "next_worker": "synthesis_worker"
        }
    else:
        return {
            "messages": [HumanMessage(content="📚 [SYSTEM]: No info found in KB")],
            "next_worker": "synthesis_worker"
        }


# ==========================================
# 9. SYNTHESIS WORKER
# ==========================================

async def synthesis_worker(state: AgentState, config: RunnableConfig):
    wf = config["configurable"]["workflow"]
    query = state["messages"][-1].content
    profile = state["user_profile"]
    
    context_parts = []
    has_sensor = state.get("sensor_data") is not None
    has_kb = state.get("kb_context") is not None

    if has_sensor:
        sensor_summary = json.dumps(state["sensor_data"], ensure_ascii=False, indent=2)
        context_parts.append(f"📋 **DỮ LIỆU CẢM BIẾN (SENSOR DATA):**\n{sensor_summary}")

    if has_kb:
        context_parts.append(f"📚 **KIẾN THỨC NÔNG NGHIỆP (KNOWLEDGE BASE):**\n{state['kb_context']}")
        
    context = "\n\n".join(context_parts)

    if not context:
        no_data_msg = AIMessage(content=(
            "Xin lỗi, tôi không tìm thấy dữ liệu hoặc thông tin liên quan đến câu hỏi này. 🌱"
        ))
        return {
            "messages": [no_data_msg],
            "next_worker": "END"
        }
        
    # Adjusted prompt based on data type
    current_time = datetime.now().strftime("%H:%M ngày %d/%m/%Y")
    user_name = profile.get('name', 'bác nông dân')
    crop_info = ', '.join(profile.get('crop_types', []))

    system_prompt = f"""Bạn là một chuyên gia tư vấn nông nghiệp AI (Kỹ sư nông nghiệp) uy tín, thân thiện tại Việt Nam.
    
THÔNG TIN NGƯỜI DÙNG:
- Tên: {user_name}
- Cây trồng: {crop_info}
- Thời gian hiện tại: {current_time}

NHIỆM VỤ:
Trả lời câu hỏi của người dùng dựa trên dữ liệu được cung cấp (Context).

YÊU CẦU QUAN TRỌNG (CHAIN OF THOUGHT):
1. **Phân tích dữ liệu:** Xem xét kỹ các chỉ số cảm biến (nếu có). Chỉ số nào bất thường (quá cao/thấp) so với tiêu chuẩn cho cây {crop_info}?
2. **Kết nối kiến thức:** Dùng thông tin từ Knowledge Base để giải thích nguyên nhân và tìm giải pháp cho các chỉ số bất thường đó.
3. **Lập luận:** Đưa ra lời khuyên dựa trên logic: Hiện trạng -> Nguyên nhân -> Giải pháp -> Hành động cụ thể.

ĐỊNH DẠNG TRẢ LỜI (TIẾNG VIỆT 100%):
- Giọng văn: Thân thiện, chuyên nghiệp, khích lệ (như một người bạn đồng hành nhà nông). Dùng từ ngữ địa phương nếu phù hợp nhưng phải dễ hiểu.
- Cấu trúc:
  1. 🌱 **Tình trạng hiện tại:** Tóm tắt ngắn gọn các chỉ số quan trọng (nêu rõ tốt hay xấu).
  2. 🔍 **Chẩn đoán/Phân tích:** Giải thích tại sao lại như vậy (dựa vào KB).
  3. 💡 **Khuyến nghị hành động:** Các bước cụ thể người dùng cần làm ngay (bón phân gì, tưới bao nhiêu, thuốc gì...).
  4. 📝 **Lưu ý:** Cảnh báo hoặc lời dặn dò thêm.

LƯU Ý: 
- Nếu thiếu dữ liệu, hãy nói rõ và gợi ý người dùng cung cấp thêm.
- Sử dụng emoji hợp lý (🌾, 🚜, 💧, ✅, ⚠️).
- Tuyệt đối không bịa đặt số liệu.

CONTEXT TỪ HỆ THỐNG:
{context}
"""
    
    messages_to_send = [SystemMessage(content=system_prompt)]
    for m in state["messages"]:
        if isinstance(m, HumanMessage) or isinstance(m, AIMessage):
            # Loại bỏ các tin nhắn chèn dữ liệu (đã ở trong System Prompt)
            # và các tin nhắn Plan/Intent meta
            if m.content and not (m.content.startswith("📋 [SYSTEM - SENSOR DATA]") or m.content.startswith("📚 [SYSTEM - KNOWLEDGE BASE]") or m.content.startswith("[")):
                 # Đảm bảo chỉ gửi nội dung, không gửi tool_calls cũ
                messages_to_send.append(m.__class__(content=m.content))

                
    # Log kiểm tra
    # print(f"📤 Sending {len(messages_to_send)} messages to LLM")

    # Stream response
    response_chunks = []
    # Sử dụng danh sách tin nhắn đã làm sạch
    async for chunk in wf.llm_smart.astream(messages_to_send): 
        if chunk.content:
            response_chunks.append(chunk.content)
    
    full_response = "".join(response_chunks)

    print(f"\n💬 Response: {full_response[:100]}...")
    
    # Self-Reflection
    if state.get("requires_reflection", True):
        reflection = await _self_reflect(wf, query, full_response, context)
        print(f"🤔 Reflection Score: {reflection.score}/100")
        
        if reflection.score < 70 and state.get("iteration", 0) < 2:
            print("🔄 Low score, regenerating...")
            system_prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{reflection.feedback}"
            
            response_chunks = []
            async for chunk in wf.llm_smart.astream([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Question: {query}")
            ]):
                if chunk.content:
                    response_chunks.append(chunk.content)
            
            full_response = "".join(response_chunks)
    
    return {
        "messages": [AIMessage(content=full_response)],
        "next_worker": "memory_update_worker",
        "iteration": state.get("iteration", 0) + 1
    }


async def _self_reflect(wf, query: str, response: str, context: str) -> ReflectionResult:
    # Prompt này giữ tiếng Anh để model GPT-4o đánh giá logic tốt hơn, 
    # nhưng thêm tiêu chí về ngôn ngữ.
    reflection_prompt = f"""Evaluate this Vietnamese agricultural AI response:

QUERY: {query}
RESPONSE: {response}
AVAILABLE CONTEXT: {context[:300]}...

CRITERIA:
1. Accuracy: Does it strictly follow the sensor data? (0-30)
2. Language: Is it natural, fluent Vietnamese suitable for farmers? (0-20)
3. Actionability: Are the recommendations specific and clear? (0-30)
4. Safety: Any harmful advice? (0-20)

Return structured evaluation."""
    
    reflector = wf.llm_fast.with_structured_output(ReflectionResult)
    result = await reflector.ainvoke(reflection_prompt)
    return result


# ==========================================
# 10. MEMORY UPDATE WORKER
# ==========================================

async def memory_update_worker(state: AgentState, config: RunnableConfig):
    wf = config["configurable"]["workflow"]
    query = state["messages"][-1].content
    
    extractor_prompt = f"""Extract user information from this query:

"{query}"

Current profile: {state["user_profile"]}

Extract ONLY NEW information.
Return structured data with only fields that have new information."""
    
    try:
        extractor = wf.llm_fast.with_structured_output(ProfileUpdate)
        profile_update = await extractor.ainvoke(extractor_prompt)
        
        updates = {
            k: v for k, v in profile_update.model_dump().items() 
            if v is not None and v != state["user_profile"].get(k)
        }
        
        if updates:
            await wf.memory_service.update_profile(state["user_id"], updates)
            print(f"💾 Updated profile: {updates}")
    
    except Exception as e:
        print(f"⚠️ Memory update failed: {e}")
    
    return {"next_worker": "END"}


# ==========================================
# 11. SUPERVISOR
# ==========================================

async def supervisor_node(state: AgentState, config: RunnableConfig):
    next_worker = state.get("next_worker", "intent_router")
    
    if next_worker == "END":
        return {"next_worker": "end"}
    
    return {"next_worker": next_worker}


# ==========================================
# 12. WORKFLOW
# ==========================================

class Workflow:
    def __init__(self, retriever, mongo_uri: str, db_name: str):
        self.llm_smart = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
        self.llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.memory_service = AsyncMemoryService(mongo_uri, db_name)
        self.kb_service = KnowledgeBaseService(retriever)
        self.kb_tool = build_kb_tool(self.kb_service)


# ==========================================
# 13. GRAPH CONSTRUCTION
# ==========================================

def build_graph(mongo_uri: str, db_name: str = "agricultural_agent"):
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("small_talk_worker", small_talk_worker)
    graph.add_node("planner_node", planner_node)  # 🔥 NEW
    graph.add_node("executor_node", executor_node)  # 🔥 NEW
    graph.add_node("retrieval_worker", retrieval_worker)
    graph.add_node("synthesis_worker", synthesis_worker)
    graph.add_node("memory_update_worker", memory_update_worker)
    
    # Entry
    graph.set_entry_point("intent_router")
    
    # Routing
    graph.add_edge("intent_router", "supervisor")
    
    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next_worker"],
        {
            "intent_router": "intent_router",
            "small_talk_worker": "small_talk_worker",
            "planner_node": "planner_node",  # 🔥 NEW
            "executor_node": "executor_node",  # 🔥 NEW
            "retrieval_worker": "retrieval_worker",
            "synthesis_worker": "synthesis_worker",
            "memory_update_worker": "memory_update_worker",
            "end": END
        }
    )
    
    # Edges
    graph.add_edge("small_talk_worker", END)
    graph.add_edge("planner_node", "supervisor")  # 🔥 NEW
    graph.add_edge("executor_node", "supervisor")  # 🔥 NEW
    graph.add_edge("retrieval_worker", "supervisor")
    graph.add_edge("synthesis_worker", "supervisor")
    graph.add_edge("memory_update_worker", END)
    
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ==========================================
# 14. MAIN FUNCTION
# ==========================================

async def run_query(query: str, user_id: str, retriever, 
                   mongo_uri: str, thread_id: str = "default"):
    workflow = Workflow(retriever, mongo_uri, "agricultural_agent")
    graph = build_graph(mongo_uri)
    profile = await workflow.memory_service.get_profile(user_id)
    
    config = {
        "configurable": {
            "thread_id": thread_id,
            "workflow": workflow
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "thread_id": thread_id,
        "user_profile": profile,
        "intent": "",
        "next_worker": "intent_router",
        "execution_plan": [],  # 🔥 NEW
        "current_step": 0,  # 🔥 NEW
        "sensor_data": None,  # 🔥 NEW
        "kb_context": None,  # 🔥 NEW
        "retrieved_docs": [],
        "should_stream": True,
        "requires_reflection": True,
        "iteration": 0
    }
    
    final_state = await graph.ainvoke(initial_state, config)
    
    ai_messages = [m for m in final_state["messages"] 
                  if isinstance(m, AIMessage) and not m.content.startswith("[")]
    final_response = ai_messages[-1].content if ai_messages else "No response"
    
    await workflow.memory_service.save_conversation(
        user_id=user_id,
        thread_id=thread_id,
        query=query,
        response=final_response,
        metadata={
            "intent": final_state.get("intent"),
            "execution_plan": final_state.get("execution_plan"),  # 🔥 NEW
            "docs_retrieved": len(final_state.get("retrieved_docs", [])),
            "iterations": final_state.get("iteration", 0)
        }
    )
    
    return final_response