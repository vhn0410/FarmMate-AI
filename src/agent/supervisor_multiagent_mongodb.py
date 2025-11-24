"""
🧠 AGRICULTURAL AI AGENT WITH LONG-TERM MEMORY
Architecture: LangGraph + MongoDB Store + LangMem

Features:
- Short-term memory: MongoDB Checkpointer (conversation history)
- Long-term memory: MongoDB Store (user preferences, facts, context)
- Memory extraction: LangMem for intelligent memory management
- Cross-thread persistence: Users remembered across sessions
"""
import asyncio
from src.tools.knowledge_base_tool import KnowledgeBaseService, build_kb_tool
from src.tools.sensorthings_tool import sensorthings_search
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
# from langraph.store.mongodb import MongoDBStore
# from langgraph.store.mongodb import MongoDBStore
from src.memory.mongo_longterm_store import MongoLongTermStore

from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
from functools import partial
import asyncio
from pymongo import MongoClient
import os

load_dotenv()

# --------------------------
# MongoDB Configuration
# --------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "agricultural_agent")

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_URI)
db = mongo_client[MONGODB_DB_NAME]

# Initialize checkpointer (short-term memory)
# checkpointer = MongoDBSaver.from_conn_string(
#     MONGODB_URI,
#     db_name=MONGODB_DB_NAME
# )

async def init_checkpointer():
    return await MongoDBSaver.from_conn_string(
        "mongodb://localhost:27017/",
        db_name="agricultural_agent"
    )

checkpointer = asyncio.run(init_checkpointer())

# Initialize store (long-term memory)
# store = MongoDBStore(
#     collection=db["agent_memory"],
#     ttl_config=None,  # No automatic expiration
#     index_config=None  # Can add vector search later
# )
store = MongoLongTermStore(collection=db["agent_memory"])

print(f"✅ MongoDB connected: {MONGODB_URI}")
print(f"✅ Database: {MONGODB_DB_NAME}")
print(f"✅ Checkpointer initialized (short-term memory)")
print(f"✅ Store initialized (long-term memory)")

# --------------------------
# State Definition
# --------------------------
def add_messages(left, right):
    """Custom reducer to prevent message duplication"""
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right

class State(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str
    enhanced_query: str
    tool_plan: dict
    sensor_data: dict
    kb_context: str
    draft_response: str
    qa_feedback: dict
    iteration_count: int
    next_agent: str
    needs_improvement: bool
    has_kb_data: bool
    has_sensor_data: bool
    conversation_stage: str
    # 🆕 LONG-TERM MEMORY FIELDS
    user_id: str  # User identifier
    user_profile: dict  # Retrieved from long-term memory
    extracted_facts: list  # Facts to store
    memory_updated: bool  # Flag if memory was updated this turn

# Agent names
SUPERVISOR = "supervisor"
QUERY_ENHANCER = "query_enhancement"
MEMORY_RETRIEVER = "memory_retrieval"
TOOL_EXECUTOR = "tool_execution"
RESPONSE_GENERATOR = "response_generation"
MEMORY_UPDATER = "memory_update"
QUALITY_ASSURER = "quality_assurance"

# --------------------------
# Memory Helper Functions
# --------------------------

async def get_user_memory(user_id: str) -> dict:
    """
    Retrieve user's long-term memory from MongoDB Store
    
    Returns:
    {
        "name": "Hoàng Vũ",
        "crop_type": "lúa",
        "location": "Cần Thơ",
        "preferences": [],
        "interaction_history": []
    }
    """
    try:
        # Namespace: ("users", user_id)
        result = await store.aget(
            namespace=("users", user_id),
            key="profile"
        )
        
        if result:
            return result.value
        else:
            # Default empty profile
            return {
                "name": None,
                "crop_type": None,
                "location": None,
                "preferences": [],
                "interaction_history": []
            }
    except Exception as e:
        print(f"[Memory] ⚠️ Error retrieving user memory: {e}")
        return {
            "name": None,
            "crop_type": None,
            "location": None,
            "preferences": [],
            "interaction_history": []
        }

async def update_user_memory(user_id: str, updates: dict):
    """
    Update user's long-term memory in MongoDB Store
    
    Args:
        user_id: User identifier
        updates: Dictionary of fields to update
    """
    try:
        # Get current memory
        current = await get_user_memory(user_id)
        
        # Merge updates
        current.update(updates)
        
        # Save back to store
        await store.aput(
            namespace=("users", user_id),
            key="profile",
            value=current
        )
        
        print(f"[Memory] ✅ Updated memory for user {user_id}")
        return True
    except Exception as e:
        print(f"[Memory] ❌ Error updating memory: {e}")
        return False

async def extract_facts_from_message(message: str, llm) -> dict:
    """
    Use LLM to extract memorable facts from user message
    
    Returns:
    {
        "name": "Hoàng Vũ",
        "crop_type": "lúa",
        "location": "Cần Thơ",
        "preferences": ["organic farming"]
    }
    """
    system_prompt = """
Bạn là memory extractor. Nhiệm vụ: trích xuất thông tin CÁ NHÂN từ câu nói của user.

TRÍCH XUẤT NẾU CÓ:
1. Tên người dùng (name)
2. Loại cây trồng (crop_type)
3. Địa điểm (location)
4. Sở thích canh tác (preferences)

OUTPUT FORMAT (JSON):
{
  "name": "Hoàng Vũ" hoặc null,
  "crop_type": "lúa" hoặc null,
  "location": "Cần Thơ" hoặc null,
  "preferences": ["organic"] hoặc []
}

VÍ DỤ:
Input: "xin chào, tôi tên là Hoàng Vũ"
Output: {"name": "Hoàng Vũ", "crop_type": null, "location": null, "preferences": []}

Input: "tôi đang trồng lúa ở Cần Thơ"
Output: {"name": null, "crop_type": "lúa", "location": "Cần Thơ", "preferences": []}

Chỉ trả về JSON, KHÔNG dùng markdown.
"""
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Message: {message}")
        ]
        result = await llm.ainvoke(messages)
        
        content = result.content.strip()
        # Remove markdown if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        facts = json.loads(content.strip())
        
        # Filter out null values
        facts = {k: v for k, v in facts.items() if v}
        
        return facts
    except Exception as e:
        print(f"[Memory Extraction] ⚠️ Error: {e}")
        return {}

# --------------------------
# Helper Functions
# --------------------------

def count_human_messages(messages: list) -> int:
    """Đếm số HumanMessage trong danh sách messages"""
    return sum(1 for m in messages if isinstance(m, HumanMessage))

def count_ai_responses(messages: list) -> int:
    """Đếm số AIMessage thực sự"""
    return sum(1 for m in messages if isinstance(m, AIMessage) and not m.content.startswith("["))

def is_new_user_turn(messages: list) -> bool:
    """Phát hiện turn mới từ user"""
    if not messages:
        return True
    
    if not isinstance(messages[-1], HumanMessage):
        return False
    
    human_count = count_human_messages(messages)
    ai_count = count_ai_responses(messages)
    
    return human_count > ai_count

# --------------------------
# Workflow class
# --------------------------
class Workflow:
    def __init__(self, retriever, llm, llm_router):
        self.retriever = retriever
        self.llm = llm
        self.llm_router = llm_router
        
        # Bind tools
        self.knowledge_base = KnowledgeBaseService(self.retriever)
        self.search_knowledge_base = build_kb_tool(self.knowledge_base)
        self.llm_with_tools = self.llm_router.bind_tools(
            tools=[self.search_knowledge_base, sensorthings_search]
        )

# --------------------------
# Agent Nodes
# --------------------------

async def supervisor(state: State, workflow: Workflow):
    """
    🧠 SUPERVISOR with Memory-Aware Routing
    """
    messages = state.get("messages", [])
    
    # ✅ Detect new user turn
    if is_new_user_turn(messages):
        latest_query = messages[-1].content
        user_id = state.get("user_id", "default_user")
        
        print(f"\n[Supervisor] 🆕 NEW TURN: '{latest_query}' from user '{user_id}'")
        
        return {
            "conversation_stage": "processing",
            "next_agent": "memory_retrieval",  # 🆕 First retrieve memory
            "iteration_count": 1,
            "original_query": latest_query,
            "enhanced_query": "",
            "tool_plan": {},
            "sensor_data": {},
            "kb_context": "",
            "draft_response": "",
            "qa_feedback": {},
            "needs_improvement": False,
            "has_kb_data": False,
            "has_sensor_data": False,
            "memory_updated": False,
            "extracted_facts": []
        }
    
    # ✅ Continue workflow
    last_agent = state.get("next_agent", "")
    iteration = state.get("iteration_count", 0)
    tool_plan = state.get("tool_plan", {})
    intent = tool_plan.get("intent", "") if tool_plan else ""
    
    # Routing logic
    if last_agent == "memory_retrieval":
        return {"next_agent": "query_enhancement"}
    
    elif last_agent == "query_enhancement":
        if intent == "small_talk":
            return {"next_agent": "small_talk_response"}
        return {"next_agent": "tool_execution"}
    
    elif last_agent == "tool_execution":
        return {"next_agent": "response_generation"}
    
    elif last_agent == "response_generation":
        # 🆕 Check if need to update memory
        if not state.get("memory_updated"):
            return {"next_agent": "memory_update"}
        elif not state.get("qa_feedback"):
            return {"next_agent": "quality_assurance"}
        else:
            return {"next_agent": "end", "conversation_stage": "completed"}
    
    elif last_agent == "memory_update":
        if not state.get("qa_feedback"):
            return {"next_agent": "quality_assurance"}
        else:
            return {"next_agent": "end", "conversation_stage": "completed"}
    
    elif last_agent == "quality_assurance":
        needs_improvement = state.get("needs_improvement", False)
        if needs_improvement and iteration < 3:
            return {"next_agent": "response_generation", "iteration_count": iteration + 1}
        return {"next_agent": "end", "conversation_stage": "completed"}
    
    elif last_agent == "small_talk_response":
        # 🆕 Update memory after small talk
        if not state.get("memory_updated"):
            return {"next_agent": "memory_update"}
        return {"next_agent": "end", "conversation_stage": "completed"}
    
    return {"next_agent": "end", "conversation_stage": "completed"}


async def memory_retrieval_agent(state: State, workflow: Workflow):
    """
    🆕 MEMORY RETRIEVAL AGENT
    Retrieve user's long-term memory before processing query
    """
    user_id = state.get("user_id", "default_user")
    
    print(f"\n[Memory Retrieval] 🔍 Loading memory for user: {user_id}")
    
    user_profile = await get_user_memory(user_id)
    
    print(f"[Memory Retrieval] 📋 Profile loaded:")
    print(f"  - Name: {user_profile.get('name', 'Unknown')}")
    print(f"  - Crop: {user_profile.get('crop_type', 'Unknown')}")
    print(f"  - Location: {user_profile.get('location', 'Unknown')}")
    
    return {
        "user_profile": user_profile,
        "messages": [AIMessage(content=f"[Memory loaded for {user_id}]")]
    }


async def query_enhancement_agent(state: State, workflow: Workflow):
    """
    QUERY ENHANCEMENT with Memory Context
    """
    original_query = state.get("original_query", "")
    user_profile = state.get("user_profile", {})
    
    # 🆕 Inject user context into prompt
    user_context = ""
    if user_profile.get("name"):
        user_context += f"User name: {user_profile['name']}\n"
    if user_profile.get("crop_type"):
        user_context += f"Crop type: {user_profile['crop_type']}\n"
    if user_profile.get("location"):
        user_context += f"Location: {user_profile['location']}\n"
    
    system = SystemMessage(content=(
        "Bạn là chuyên gia phân tích ý định người dùng.\n\n"
        f"{'THÔNG TIN USER:\n' + user_context if user_context else ''}\n"
        "# NHIỆM VỤ: Phân tích câu hỏi và xác định intent + data_needs\n\n"
        "# INTENT TYPES:\n"
        "- small_talk: Chào hỏi, giới thiệu, cảm ơn\n"
        "- check_sensor_data: Kiểm tra dữ liệu cảm biến\n"
        "- ask_knowledge: Hỏi kiến thức nông nghiệp\n"
        "- consultation: Tư vấn giải quyết vấn đề\n\n"
        "# DATA NEEDS:\n"
        "- []: Không cần tool (small_talk)\n"
        "- ['sensor_data']: Cần dữ liệu sensor\n"
        "- ['knowledge_base']: Cần KB\n"
        "- ['sensor_data', 'knowledge_base']: Cần cả hai\n\n"
        "OUTPUT JSON (không markdown):\n"
        "{\"intent\": \"...\", \"enhanced_query\": \"...\", \"data_needs\": [...], \"context\": \"...\"}"
    ))
    
    messages = [system, HumanMessage(content=original_query)]
    result = await workflow.llm_router.ainvoke(messages)
    
    try:
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        enhanced = json.loads(content.strip())
    except Exception as e:
        print(f"[Query Enhancement] Parse error: {e}")
        enhanced = {
            "intent": "ask_knowledge",
            "enhanced_query": original_query,
            "data_needs": ["knowledge_base"],
            "context": ""
        }
    
    print(f"[Query Enhancement] Intent: {enhanced.get('intent')}")
    
    return {
        "enhanced_query": enhanced.get("enhanced_query", original_query),
        "tool_plan": enhanced,
        "messages": [AIMessage(content=f"[Enhanced Query]")]
    }


async def tool_execution_agent(state: State, workflow: Workflow):
    """TOOL EXECUTION (unchanged)"""
    tool_plan = state.get("tool_plan", {})
    data_needs = tool_plan.get("data_needs", [])
    enhanced_query = state.get("enhanced_query", state.get("original_query", ""))
    
    if tool_plan.get("intent") == "small_talk" or not data_needs:
        return {
            "sensor_data": {},
            "kb_context": "",
            "has_kb_data": False,
            "has_sensor_data": False,
            "messages": [ToolMessage(content="Skipped", tool_call_id="none")]
        }
    
    sensor_data = {}
    kb_context = ""
    tool_messages = []
    has_kb_data = False
    has_sensor_data = False
    
    if "sensor_data" in data_needs:
        try:
            sensor_result = await sensorthings_search.ainvoke(enhanced_query)
            if isinstance(sensor_result, dict) and sensor_result:
                sensor_data.update(sensor_result)
                has_sensor_data = True
                tool_messages.append(ToolMessage(
                    content=json.dumps(sensor_result, ensure_ascii=False, indent=2),
                    tool_call_id="sensor_tool"
                ))
        except Exception as e:
            tool_messages.append(ToolMessage(
                content=f"Sensor error: {e}",
                tool_call_id="sensor_tool"
            ))
    
    if "knowledge_base" in data_needs:
        try:
            kb_result = await workflow.search_knowledge_base.ainvoke(enhanced_query)
            if kb_result and len(str(kb_result).strip()) > 10:
                kb_context = kb_result
                has_kb_data = True
                tool_messages.append(ToolMessage(
                    content=kb_result,
                    tool_call_id="kb_tool"
                ))
        except Exception as e:
            tool_messages.append(ToolMessage(
                content=f"KB error: {e}",
                tool_call_id="kb_tool"
            ))
    
    return {
        "sensor_data": sensor_data,
        "kb_context": kb_context,
        "has_kb_data": has_kb_data,
        "has_sensor_data": has_sensor_data,
        "messages": tool_messages
    }


async def small_talk_response_agent(state: State, workflow: Workflow):
    """
    🆕 SMALL TALK with Memory Personalization
    """
    original_query = state.get("original_query", "").lower()
    user_profile = state.get("user_profile", {})
    user_name = user_profile.get("name")
    
    # Greeting
    if any(greet in original_query for greet in ["xin chào", "hello", "hi", "chào"]):
        if user_name:
            response = (
                f"Xin chào {user_name}! 👋 Rất vui được gặp lại bạn.\n\n"
                "Tôi có thể giúp bạn:\n"
                "📊 Kiểm tra dữ liệu cảm biến\n"
                "📚 Tư vấn kỹ thuật canh tác\n"
                "💡 Giải đáp thắc mắc\n\n"
                "Bạn cần hỗ trợ gì không?"
            )
        else:
            response = (
                "Xin chào! 👋 Tôi là trợ lý nông nghiệp.\n\n"
                "Tôi có thể giúp bạn với:\n"
                "📊 Dữ liệu cảm biến\n"
                "📚 Kỹ thuật canh tác\n"
                "💡 Tư vấn chuyên sâu\n\n"
                "Bạn cần gì không?"
            )
    
    # Self-introduction
    elif any(intro in original_query for intro in ["tôi là", "mình là", "tên tôi", "tên mình"]):
        response = (
            "Rất vui được làm quen! 🌾\n\n"
            "Tôi sẽ nhớ thông tin của bạn để hỗ trợ tốt hơn trong tương lai.\n"
            "Bạn đang trồng loại cây gì và cần tư vấn về vấn đề gì không?"
        )
    
    # Identity question
    elif any(who in original_query for who in ["bạn là ai", "tên bạn"]):
        response = (
            "Tôi là trợ lý nông nghiệp AI! 🤖🌾\n\n"
            "Chuyên môn:\n"
            "✅ Phân tích dữ liệu cảm biến\n"
            "✅ Tư vấn canh tác\n"
            "✅ Giải đáp kỹ thuật\n\n"
            "Bạn cần hỗ trợ gì?"
        )
    
    # "Who am I"
    elif "tôi là ai" in original_query:
        if user_name:
            response = f"Bạn là {user_name}! 😊\n\n"
            if user_profile.get("crop_type"):
                response += f"Bạn đang trồng {user_profile['crop_type']}"
                if user_profile.get("location"):
                    response += f" ở {user_profile['location']}"
                response += ".\n\n"
            response += "Tôi có thể giúp gì thêm không?"
        else:
            response = (
                "Tôi chưa biết tên bạn. 🤔\n\n"
                "Bạn có thể tự giới thiệu để tôi nhớ và hỗ trợ tốt hơn nhé!"
            )
    
    # Thanks
    elif any(thanks in original_query for thanks in ["cảm ơn", "thanks"]):
        response = (
            "Không có gì! 😊\n\n"
            "Rất vui được hỗ trợ bạn. Hãy quay lại nếu cần thêm giúp đỡ nhé! 🌱"
        )
    
    # Goodbye
    elif any(bye in original_query for bye in ["bye", "tạm biệt"]):
        response = (
            "Tạm biệt! 👋 Chúc bạn mùa màng bội thu! 🌾\n\n"
            "Hẹn gặp lại!"
        )
    
    else:
        response = (
            "Tôi là trợ lý nông nghiệp. Tôi có thể giúp bạn:\n"
            "- Kiểm tra dữ liệu cảm biến\n"
            "- Tư vấn kỹ thuật\n"
            "- Chăm sóc cây trồng\n\n"
            "Bạn cần gì?"
        )
    
    return {
        "draft_response": response,
        "messages": [AIMessage(content=response)]
    }


async def response_generation_agent(state: State, workflow: Workflow):
    """
    🆕 RESPONSE GENERATION with Memory Context
    """
    has_sensor_data = state.get("has_sensor_data", False)
    has_kb_data = state.get("has_kb_data", False)
    sensor_data = state.get("sensor_data", {})
    kb_context = state.get("kb_context", "")
    original_query = state.get("original_query", "")
    user_profile = state.get("user_profile", {})
    
    # Build user context
    user_context = ""
    if user_profile.get("name"):
        user_context += f"User: {user_profile['name']}\n"
    if user_profile.get("crop_type"):
        user_context += f"Crop: {user_profile['crop_type']}\n"
    if user_profile.get("location"):
        user_context += f"Location: {user_profile['location']}\n"
    
    if not has_kb_data and not has_sensor_data:
        no_data_response = AIMessage(content=(
            "Xin lỗi, tôi không tìm thấy thông tin về câu hỏi này. 🤔\n\n"
            "Hãy hỏi về:\n"
            "- Kỹ thuật trồng trọt 🌱\n"
            "- Dữ liệu cảm biến 📊\n"
            "- Chăm sóc cây trồng 💧"
        ))
        return {
            "draft_response": no_data_response.content,
            "messages": [no_data_response]
        }
    
    # Build system prompt with user context
    if has_sensor_data and has_kb_data:
        system_prompt = (
            f"Bạn là chuyên gia nông nghiệp.\n\n"
            f"{user_context}\n"
            "LUẬT:\n"
            "- CHỈ dùng thông tin từ Sensor Data và KB\n"
            "- KHÔNG bịa thêm\n"
            "- Nếu thiếu → nói rõ\n\n"
            "NHIỆM VỤ:\n"
            "1. Phân tích sensor (so với ngưỡng từ KB)\n"
            "2. Đưa ra khuyến nghị cụ thể\n"
            "3. Ngắn gọn, dễ hiểu, có emoji\n"
        )
        data_context = f"\n\n--- SENSOR ---\n{json.dumps(sensor_data, ensure_ascii=False, indent=2)}\n\n--- KB ---\n{kb_context}"
    
    elif has_kb_data:
        system_prompt = (
            f"Bạn là chuyên gia nông nghiệp.\n\n"
            f"{user_context}\n"
            "CHỈ dùng thông tin từ KB. Trả lời ngắn gọn, có emoji."
        )
        data_context = f"\n\n--- KB ---\n{kb_context}"
    
    else:
        system_prompt = (
            f"Bạn là chuyên gia nông nghiệp.\n\n"
            f"{user_context}\n"
            "CHỈ mô tả dữ liệu sensor. Không đưa ra khuyến nghị nếu thiếu KB."
        )
        data_context = f"\n\n--- SENSOR ---\n{json.dumps(sensor_data, ensure_ascii=False, indent=2)}"
    
    system = SystemMessage(content=system_prompt)
    human = HumanMessage(content=f"Câu hỏi: {original_query}{data_context}")
    
    messages = [system, human]
    
    try:
        result = await workflow.llm.ainvoke(messages)
        
        return {
            "draft_response": result.content,
            "messages": [result]
        }
    except Exception as e:
        error_response = AIMessage(content=f"Xin lỗi, lỗi: {str(e)}")
        return {
            "draft_response": error_response.content,
            "messages": [error_response]
        }


async def memory_update_agent(state: State, workflow: Workflow):
    """
    🆕 MEMORY UPDATE AGENT
    Extract and save facts from current conversation turn
    """
    original_query = state.get("original_query", "")
    user_id = state.get("user_id", "default_user")
    
    print(f"\n[Memory Update] 🧠 Extracting facts from: '{original_query}'")
    
    # Extract facts using LLM
    extracted_facts = await extract_facts_from_message(original_query, workflow.llm_router)
    
    if extracted_facts:
        print(f"[Memory Update] 📝 Extracted: {extracted_facts}")
        
        # Update long-term memory
        success = await update_user_memory(user_id, extracted_facts)
        
        if success:
            return {
                "extracted_facts": [extracted_facts],
                "memory_updated": True,
                "messages": [AIMessage(content=f"[Memory updated: {list(extracted_facts.keys())}]")]
            }
    
    print(f"[Memory Update] ℹ️ No new facts to store")
    return {
        "memory_updated": True,
        "messages": [AIMessage(content="[No memory updates]")]
    }


async def quality_assurance_agent(state: State, workflow: Workflow):
    """QUALITY ASSURANCE (unchanged)"""
    system = SystemMessage(content=(
        "Bạn là QA Agent - kiểm tra chất lượng câu trả lời.\n\n"
        "TIÊU CHÍ:\n"
        "1. ✅ Chính xác: CHỈ dùng info từ tools\n"
        "2. ✅ Đầy đủ: Trả lời đủ câu hỏi\n"
        "3. ✅ Hữu ích: Có khuyến nghị cụ thể\n"
        "4. ✅ Trung thực: Nếu thiếu → nói rõ\n\n"
        "OUTPUT JSON:\n"
        "{\n"
        "  \"score\": 0-100,\n"
        "  \"needs_improvement\": true/false,\n"
        "  \"strengths\": [\"...\"],\n"
        "  \"weaknesses\": [\"...\"],\n"
        "  \"suggestions\": \"...\"\n"
        "}\n"
    ))
    
    draft_response = state.get("draft_response", "")
    original_query = state.get("original_query", "")
    
    evaluation_prompt = f"""
CÂU HỎI: {original_query}
CÂU TRẢ LỜI: {draft_response}

Đánh giá và trả về JSON:
"""
    
    messages = [system, HumanMessage(content=evaluation_prompt)]
    result = await workflow.llm_router.ainvoke(messages)
    
    try:
        content = result.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])
            if content.startswith("json"):
                content = content[4:].strip()
        
        qa_feedback = json.loads(content.strip())
    except Exception as e:
        qa_feedback = {
            "score": 85,
            "needs_improvement": False,
            "strengths": ["OK"],
            "weaknesses": [],
            "suggestions": ""
        }
    
    needs_improvement = qa_feedback.get("needs_improvement", False)
    
    print(f"[QA] Score: {qa_feedback.get('score')}/100")
    
    return {
        "qa_feedback": qa_feedback,
        "needs_improvement": needs_improvement,
        "messages": [AIMessage(content=f"[QA: {qa_feedback.get('score')}/100]")]
    }


# --------------------------
# Router Functions
# --------------------------

def supervisor_router(state: State) -> str:
    """Route based on supervisor decision"""
    next_agent = state.get("next_agent", "end")
    
    if next_agent == "end":
        return "end"
    
    return next_agent


# --------------------------
# Build Graph
# --------------------------
def build_graph(workflow: Workflow):
    """
    🧠 Build LangGraph with Long-term Memory
    """
    graph_builder = StateGraph(State)
    
    # Add all nodes
    graph_builder.add_node("supervisor", partial(supervisor, workflow=workflow))
    graph_builder.add_node("memory_retrieval", partial(memory_retrieval_agent, workflow=workflow))
    graph_builder.add_node("query_enhancement", partial(query_enhancement_agent, workflow=workflow))
    graph_builder.add_node("tool_execution", partial(tool_execution_agent, workflow=workflow))
    graph_builder.add_node("response_generation", partial(response_generation_agent, workflow=workflow))
    graph_builder.add_node("memory_update", partial(memory_update_agent, workflow=workflow))
    graph_builder.add_node("quality_assurance", partial(quality_assurance_agent, workflow=workflow))
    graph_builder.add_node("small_talk_response", partial(small_talk_response_agent, workflow=workflow))
    
    # Set entry point
    graph_builder.set_entry_point("supervisor")
    
    # Add conditional routing
    graph_builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "memory_retrieval": "memory_retrieval",
            "query_enhancement": "query_enhancement",
            "tool_execution": "tool_execution",
            "response_generation": "response_generation",
            "memory_update": "memory_update",
            "quality_assurance": "quality_assurance",
            "small_talk_response": "small_talk_response",
            "end": END
        }
    )
    
    # All nodes return to supervisor
    graph_builder.add_edge("memory_retrieval", "supervisor")
    graph_builder.add_edge("query_enhancement", "supervisor")
    graph_builder.add_edge("tool_execution", "supervisor")
    graph_builder.add_edge("response_generation", "supervisor")
    graph_builder.add_edge("memory_update", "supervisor")
    graph_builder.add_edge("quality_assurance", "supervisor")
    graph_builder.add_edge("small_talk_response", "supervisor")
    
    return graph_builder.compile(checkpointer=checkpointer)


# --------------------------
# Usage Example
# --------------------------
async def run_agent(query: str, retriever, thread_id: str = "default", user_id: str = "default_user"):
    """
    Run agent with memory support
    
    Args:
        query: User's question
        retriever: Knowledge base retriever
        thread_id: Conversation thread ID (for short-term memory)
        user_id: User identifier (for long-term memory)
    """
    workflow = Workflow(retriever, 
                       ChatOpenAI(model="gpt-4o", temperature=0, streaming=True),
                       ChatOpenAI(model="gpt-4o-mini", temperature=0))
    graph = build_graph(workflow)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "iteration_count": 0,
        "conversation_stage": "new_turn",
        "user_id": user_id  # 🆕 Pass user ID
    }
    
    final_state = await graph.ainvoke(initial_state, config)
    
    return final_state["messages"][-1].content


# --------------------------
# Testing Functions
# --------------------------

async def test_memory():
    """
    Test long-term memory functionality
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    
    # Mock retriever
    docs = [Document(page_content="Test doc")]
    bm25 = BM25Retriever.from_documents(docs)
    
    print("="*60)
    print("🧪 TESTING LONG-TERM MEMORY")
    print("="*60)
    
    # Test 1: First conversation
    print("\n--- Test 1: User introduces themselves ---")
    response1 = await run_agent(
        query="xin chào, tôi tên là Hoàng Vũ",
        retriever=bm25,
        thread_id="test_thread_1",
        user_id="user_hoang_vu"
    )
    print(f"Response: {response1}\n")
    
    # Test 2: Add more info
    print("--- Test 2: User adds crop info ---")
    response2 = await run_agent(
        query="tôi đang trồng lúa ở Cần Thơ",
        retriever=bm25,
        thread_id="test_thread_1",
        user_id="user_hoang_vu"
    )
    print(f"Response: {response2}\n")
    
    # Test 3: Check memory (same thread)
    print("--- Test 3: Ask 'who am I' (same thread) ---")
    response3 = await run_agent(
        query="tôi là ai?",
        retriever=bm25,
        thread_id="test_thread_1",
        user_id="user_hoang_vu"
    )
    print(f"Response: {response3}\n")
    
    # Test 4: NEW THREAD - Check long-term memory
    print("--- Test 4: NEW THREAD - Memory should persist ---")
    response4 = await run_agent(
        query="tôi là ai?",
        retriever=bm25,
        thread_id="test_thread_2",  # Different thread
        user_id="user_hoang_vu"  # Same user
    )
    print(f"Response: {response4}\n")
    
    # Test 5: Check what I'm growing
    print("--- Test 5: Ask what I'm growing (new thread) ---")
    response5 = await run_agent(
        query="tôi đang trồng gì?",
        retriever=bm25,
        thread_id="test_thread_3",  # Another new thread
        user_id="user_hoang_vu"
    )
    print(f"Response: {response5}\n")
    
    print("="*60)
    print("✅ MEMORY TEST COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run test
    import asyncio
    asyncio.run(test_memory())