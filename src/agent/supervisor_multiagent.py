from src.tools.knowledge_base_tool import KnowledgeBaseService, build_kb_tool
from src.tools.sensorthings_tool import sensorthings_search
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json
from functools import partial
import asyncio
load_dotenv()

# --------------------------
# Memory & Graph State
# --------------------------
memory = MemorySaver()

def add_messages(left, right):
    """Custom reducer to prevent message duplication"""
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right

class State(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str  # Câu hỏi gốc
    enhanced_query: str  # Câu hỏi đã enhance
    tool_plan: dict  # Kế hoạch gọi tools
    sensor_data: dict  # Dữ liệu cảm biến
    kb_context: str  # Context từ KB
    draft_response: str  # Câu trả lời nháp
    qa_feedback: dict  # Feedback từ QA agent
    iteration_count: int  # Số lần iteration
    next_agent: str  # Agent tiếp theo
    needs_improvement: bool  # Flag để quyết định có cần cải thiện không
    has_kb_data: bool  # Flag kiểm tra có dữ liệu KB không
    has_sensor_data: bool  # Flag kiểm tra có dữ liệu sensor không

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
# LLM setup
# --------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------------
# Agent Nodes
# --------------------------

async def supervisor(state: State, workflow: Workflow):
    """
    Supervisor Agent - Điều phối workflow
    Quyết định agent nào chạy tiếp theo
    """
    iteration = state.get("iteration_count", 0)
    
    # Nếu người dùng chỉ chat xã giao, kết thúc luôn
    # Luồng chính: Enhancement → Tool Execution → Response → QA
    if iteration == 0:
        return {"next_agent": "query_enhancement", "iteration_count": 1}
    
    last_agent = state.get("next_agent", "")
    
    if last_agent == "query_enhancement" and state.get("tool_plan", {}).get("intent") == "small_talk":
        return {"next_agent": "end"}
    
    if last_agent == "query_enhancement":
        return {"next_agent": "tool_execution"}
    
    elif last_agent == "tool_execution":
        return {"next_agent": "response_generation"}
    
    elif last_agent == "response_generation":
        # Nếu chưa qua QA lần nào, bắt buộc phải qua QA
        if not state.get("qa_feedback"):
            return {"next_agent": "quality_assurance"}
        # Nếu đã qua QA và không cần cải thiện nữa
        else:
            return {"next_agent": "end"}
    
    elif last_agent == "quality_assurance":
        needs_improvement = state.get("needs_improvement", False)
        
        # Nếu cần cải thiện và chưa quá 2 lần
        if needs_improvement and iteration < 3:
            return {
                "next_agent": "response_generation",
                "iteration_count": iteration + 1
            }
        else:
            return {"next_agent": "end"}
    
    return {"next_agent": "end"}


async def query_enhancement_agent(state: State, workflow: Workflow):
    """
    Query Enhancement Agent
    - Phân tích ý định người dùng
    - Mở rộng câu hỏi với context nông nghiệp
    - Xác định loại thông tin cần thiết
    """
    system = SystemMessage(content=(
        "Bạn là Query Enhancement Agent - chuyên gia phân tích câu hỏi nông nghiệp.\n\n"
        "NHIỆM VỤ:\n"
        "1. Phân tích ý định của nông dân:\n"
        "   - 'check_sensor_data': Kiểm tra dữ liệu thực tế (nhiệt độ, độ ẩm, NPK, pH, cảm biến)\n"
        "   - 'ask_knowledge': Hỏi kiến thức chung (kỹ thuật, phương pháp, bệnh hại)\n"
        "   - 'consultation': Cần tư vấn (bón phân, xử lý bệnh)\n"
        "2. Mở rộng câu hỏi với thuật ngữ chuyên môn\n"
        "3. Xác định tools cần gọi\n\n"
        "LUẬT QUAN TRỌNG:\n"
        "- Nếu câu hỏi về KIỂM TRA/THEO DÕI dữ liệu → data_needs: ['sensor_data', 'knowledge_base']\n"
        "- Nếu câu hỏi về KIẾN THỨC/HƯỚNG DẪN → data_needs: ['knowledge_base']\n"
        "- LUÔN phải có ít nhất 1 tool trong data_needs\n\n"
        "VÍ DỤ:\n"
        "Input: 'Kiểm tra đất'\n"
        "Output: {\n"
        "  'intent': 'check_sensor_data',\n"
        "  'enhanced_query': 'Kiểm tra các chỉ số đất: NPK, pH, độ ẩm, nhiệt độ đất',\n"
        "  'data_needs': ['sensor_data', 'knowledge_base'],\n"
        "  'context': 'Nông dân cần biết tình trạng đất để quyết định bón phân'\n"
        "}\n\n"
        "Input: 'Cách bón phân cho lúa'\n"
        "Output: {\n"
        "  'intent': 'ask_knowledge',\n"
        "  'enhanced_query': 'Hướng dẫn bón phân cho lúa: loại phân, liều lượng, thời điểm',\n"
        "  'data_needs': ['knowledge_base'],\n"
        "  'context': 'Nông dân cần hướng dẫn kỹ thuật bón phân'\n"
        "}\n\n"
        "Trả về JSON format. Phân tích câu hỏi sau:"
    ))
    
    original_query = state["messages"][-1].content if state["messages"] else state.get("original_query", "")
    
    messages = [system, HumanMessage(content=original_query)]
    result = await workflow.llm_router.ainvoke(messages)
    
    # Nhận diện câu đơn giản, chào hỏi, không liên quan nông nghiệp
    if any(greet in original_query.lower() for greet in ["xin chào", "hello", "hi", "chào bạn"]):
        enhanced = {
            "intent": "small_talk",
            "enhanced_query": original_query,
            "data_needs": [],  # Không cần gọi tool
            "context": "Người dùng chỉ chào hỏi, không yêu cầu thông tin nông nghiệp"
        }
    else:
        # Parse enhanced query
        try:
            content = result.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            enhanced = json.loads(content.strip())
        except Exception as e:
            print(f"[Query Enhancement] Parse error: {e}, using default")
            enhanced = {
                "intent": "ask_knowledge",
                "enhanced_query": original_query,
                "data_needs": ["knowledge_base"],
                "context": ""
            }
        
    return {
        "original_query": original_query,
        "enhanced_query": enhanced.get("enhanced_query", original_query),
        "tool_plan": enhanced,
        "messages": [AIMessage(content=f"[Enhanced Query: {enhanced.get('enhanced_query')}]")]
    }


async def tool_execution_agent(state: State, workflow: Workflow):
    """
    Tool Execution Agent - BẮT BUỘC gọi tools
    - Gọi tools dựa trên tool_plan
    - Thu thập dữ liệu từ sensors và knowledge base
    """
    tool_plan = state.get("tool_plan", {})
    data_needs = tool_plan.get("data_needs", ["knowledge_base"])
    enhanced_query = state.get("enhanced_query", state.get("original_query", ""))
    
    sensor_data = {}
    kb_context = ""
    tool_messages = []
    has_kb_data = False
    has_sensor_data = False
    
    print(f"\n[Tool Execution] Query: {enhanced_query}")
    print(f"[Tool Execution] Data needs: {data_needs}")
    
    # 1. GỌI SENSOR TOOL (nếu cần)
    if "sensor_data" in data_needs or "soil_sensors" in data_needs:
        print(f"[Tool Execution] Calling sensorthings_search...")
        try:
            sensor_result = await sensorthings_search.ainvoke(enhanced_query)
            print(f"[Tool Execution] Sensor result type: {type(sensor_result)}")
            
            if isinstance(sensor_result, dict) and sensor_result:
                sensor_data.update(sensor_result)
                has_sensor_data = True
                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(sensor_result, ensure_ascii=False, indent=2),
                        tool_call_id="sensor_tool"
                    )
                )
                print(f"[Tool Execution] ✅ Got sensor data: {len(sensor_data)} keys")
            else:
                tool_messages.append(
                    ToolMessage(content="Không có dữ liệu cảm biến", tool_call_id="sensor_tool")
                )
                print(f"[Tool Execution] ⚠️ No sensor data")
        except Exception as e:
            error_msg = f"Lỗi khi lấy dữ liệu cảm biến: {str(e)}"
            tool_messages.append(
                ToolMessage(content=error_msg, tool_call_id="sensor_tool")
            )
            print(f"[Tool Execution] ❌ Sensor error: {e}")
    
    # 2. GỌI KNOWLEDGE BASE TOOL (bắt buộc)
    print(f"[Tool Execution] Calling search_knowledge_base...")
    try:
        kb_result = await workflow.search_knowledge_base.ainvoke(enhanced_query)
        print(f"[Tool Execution] KB result length: {len(str(kb_result))}")
        
        if kb_result and len(str(kb_result).strip()) > 10:
            kb_context = kb_result
            has_kb_data = True
            tool_messages.append(
                ToolMessage(content=kb_result, tool_call_id="kb_tool")
            )
            print(f"[Tool Execution] ✅ Got KB data")
        else:
            kb_context = "Không tìm thấy thông tin liên quan trong cơ sở kiến thức."
            tool_messages.append(
                ToolMessage(content=kb_context, tool_call_id="kb_tool")
            )
            print(f"[Tool Execution] ⚠️ No KB data")
    except Exception as e:
        error_msg = f"Lỗi khi tìm kiếm knowledge base: {str(e)}"
        kb_context = error_msg
        tool_messages.append(
            ToolMessage(content=error_msg, tool_call_id="kb_tool")
        )
        print(f"[Tool Execution] ❌ KB error: {e}")
    
    return {
        "sensor_data": sensor_data,
        "kb_context": kb_context,
        "has_kb_data": has_kb_data,
        "has_sensor_data": has_sensor_data,
        "messages": tool_messages
    }


async def response_generation_agent(state: State, workflow: Workflow):
    """
    Response Generation Agent (STREAMING ENABLED)
    - CHỈ trả lời dựa trên dữ liệu từ tools
    - Nếu không có dữ liệu → nói không biết
    
    FIX: Không truyền ToolMessage vào LLM, chỉ truyền data dạng text
    """
    has_sensor_data = state.get("has_sensor_data", False)
    has_kb_data = state.get("has_kb_data", False)
    sensor_data = state.get("sensor_data", {})
    kb_context = state.get("kb_context", "")
    qa_feedback = state.get("qa_feedback", {})
    original_query = state.get("original_query", "")
    
    # KIỂM TRA có dữ liệu không
    if not has_kb_data and not has_sensor_data:
        # KHÔNG CÓ DỮ LIỆU → Trả lời không biết
        no_data_response = AIMessage(content=(
            "Xin lỗi, tôi không tìm thấy thông tin về câu hỏi này trong cơ sở dữ liệu. "
            "Vui lòng hỏi các câu hỏi liên quan đến:\n"
            "- Kỹ thuật trồng trọt\n"
            "- Bón phân và chăm sóc cây\n"
            "- Kiểm tra dữ liệu cảm biến (nhiệt độ, độ ẩm, NPK, pH)"
        ))
        return {
            "draft_response": no_data_response.content,
            "messages": [no_data_response]
        }
    
    # CÓ DỮ LIỆU → Tạo prompt
    if has_sensor_data and has_kb_data:
        system_prompt = (
            "Bạn là chuyên gia nông nghiệp. CHỈ trả lời dựa trên dữ liệu được cung cấp.\n\n"
            "LUẬT QUAN TRỌNG:\n"
            "- CHỈ sử dụng thông tin từ Sensor Data và Knowledge Base Context bên dưới\n"
            "- KHÔNG bịa thêm thông tin không có trong dữ liệu\n"
            "- Nếu thiếu thông tin → nói rõ 'cần thêm dữ liệu về...'\n\n"
            "NHIỆM VỤ:\n"
            "1. PHÂN TÍCH DỮ LIỆU SENSOR:\n"
            "   - So sánh với ngưỡng chuẩn từ KB\n"
            "   - Đánh giá: tốt ✅ / cần cải thiện ⚠️ / nguy hiểm ❌\n"
            "   - Giải thích nguyên nhân\n\n"
            "2. KHUYẾN NGHỊ CỤ THỂ:\n"
            "   - Dựa trên kiến thức từ KB\n"
            "   - Tên sản phẩm, liều lượng, cách dùng\n"
            "   - Thời gian thực hiện\n\n"
            "3. ĐỊNH DẠNG:\n"
            "   - Dùng emoji\n"
            "   - Ngắn gọn, dễ hiểu\n"
            "   - Hành động cụ thể\n"
        )
        data_context = f"\n\n--- SENSOR DATA ---\n{json.dumps(sensor_data, ensure_ascii=False, indent=2)}\n\n--- KNOWLEDGE BASE ---\n{kb_context}"
    
    elif has_kb_data:
        system_prompt = (
            "Bạn là chuyên gia nông nghiệp. CHỈ trả lời dựa trên Knowledge Base Context được cung cấp.\n\n"
            "LUẬT QUAN TRỌNG:\n"
            "- CHỈ sử dụng thông tin từ Knowledge Base Context bên dưới\n"
            "- KHÔNG bịa thêm thông tin không có trong KB\n"
            "- Nếu KB không có đủ thông tin → nói rõ 'không có thông tin về...'\n\n"
            "NHIỆM VỤ:\n"
            "1. Trả lời dựa trên KB\n"
            "2. Đưa ra các bước cụ thể\n"
            "3. Số liệu chính xác (nếu có)\n"
            "4. Ngắn gọn, dễ hiểu\n"
        )
        data_context = f"\n\n--- KNOWLEDGE BASE ---\n{kb_context}"
    
    else:  # has_sensor_data only
        system_prompt = (
            "Bạn là chuyên gia nông nghiệp. CHỈ trả lời dựa trên Sensor Data được cung cấp.\n\n"
            "LUẬT QUAN TRỌNG:\n"
            "- CHỈ mô tả dữ liệu sensor\n"
            "- KHÔNG đưa ra khuyến nghị nếu không có KB context\n"
            "- Nói rõ 'cần thêm thông tin về ngưỡng chuẩn'\n"
        )
        data_context = f"\n\n--- SENSOR DATA ---\n{json.dumps(sensor_data, ensure_ascii=False, indent=2)}"
    
    # Nếu có feedback từ QA
    if qa_feedback.get("needs_improvement"):
        system_prompt += f"\n\nCẢI THIỆN DỰA TRÊN FEEDBACK:\n{qa_feedback.get('suggestions', '')}"
    
    # QUAN TRỌNG: Chỉ dùng SystemMessage và HumanMessage
    # KHÔNG dùng ToolMessage vì không có AIMessage với tool_calls đứng trước
    system = SystemMessage(content=system_prompt)
    human = HumanMessage(content=f"Câu hỏi: {original_query}{data_context}")
    
    messages = [system, human]
    
    # STREAM response
    try:
        result = await workflow.llm.ainvoke(messages)
        
        return {
            "draft_response": result.content,
            "messages": [result]
        }
    except asyncio.CancelledError:
        # Khi bị hủy bởi LangGraph/LangSmith, ta không cần xem là lỗi
        print("Responder task was cancelled (likely by LangGraph).")
        return None


async def quality_assurance_agent(state: State, workflow: Workflow):
    """
    Quality Assurance Agent
    - Kiểm tra có trả lời đúng dựa trên dữ liệu không
    - Có bịa thông tin không
    """
    system = SystemMessage(content=(
        "Bạn là QA Agent - kiểm tra chất lượng câu trả lời.\n\n"
        "TIÊU CHÍ QUAN TRỌNG:\n"
        "1. ✅ Chính xác: CHỈ dùng thông tin từ tools, KHÔNG bịa\n"
        "2. ✅ Đầy đủ: Trả lời đủ câu hỏi\n"
        "3. ✅ Hữu ích: Có khuyến nghị cụ thể (nếu có KB data)\n"
        "4. ✅ Trung thực: Nếu thiếu data → phải nói rõ\n\n"
        "CẢI THIỆN NẾU:\n"
        "- Score < 75\n"
        "- Có thông tin không có trong tool results\n"
        "- Thiếu phân tích quan trọng\n\n"
        "OUTPUT FORMAT:\n"
        "{\n"
        "  'score': 0-100,\n"
        "  'needs_improvement': true/false,\n"
        "  'strengths': ['điểm mạnh 1'],\n"
        "  'weaknesses': ['điểm yếu 1'],\n"
        "  'suggestions': 'Cải thiện cụ thể...'\n"
        "}\n"
    ))
    
    draft_response = state.get("draft_response", "")
    original_query = state.get("original_query", "")
    has_sensor_data = state.get("has_sensor_data", False)
    has_kb_data = state.get("has_kb_data", False)
    kb_context = state.get("kb_context", "")[:500]  # First 500 chars
    
    evaluation_prompt = f"""
CÂU HỎI: {original_query}

DỮ LIỆU CÓ:
- Sensor: {'Có' if has_sensor_data else 'Không'}
- KB: {'Có' if has_kb_data else 'Không'}
- KB Preview: {kb_context if has_kb_data else 'N/A'}

CÂU TRẢ LỜI:
{draft_response}

Đánh giá và trả về JSON:
"""
    
    messages = [system, HumanMessage(content=evaluation_prompt)]
    result = await workflow.llm_router.ainvoke(messages)
    
    # Parse QA feedback (ROBUST PARSING)
    try:
        content = result.content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])  # Remove first and last lines
            if content.startswith("json"):
                content = content[4:].strip()
        
        # Replace single quotes with double quotes for JSON compatibility
        content = content.replace("'", '"')
        
        qa_feedback = json.loads(content.strip())
    except Exception as e:
        print(f"[QA] Parse error: {e}")
        print(f"[QA] Raw content: {result.content[:200]}")
        qa_feedback = {
            "score": 85,
            "needs_improvement": False,
            "strengths": ["Câu trả lời hợp lý"],
            "weaknesses": [],
            "suggestions": ""
        }
    
    needs_improvement = qa_feedback.get("needs_improvement", False)
    
    return {
        "qa_feedback": qa_feedback,
        "needs_improvement": needs_improvement,
        "messages": [AIMessage(content=f"[QA Score: {qa_feedback.get('score')}/100]")]
    }


# --------------------------
# Router Functions
# --------------------------

def supervisor_router(state: State) -> str:
    """Route dựa trên quyết định của supervisor"""
    next_agent = state.get("next_agent", "end")
    
    if next_agent == "end":
        return "end"
    
    return next_agent


# --------------------------
# Build Graph
# --------------------------
def build_graph(workflow: Workflow):
    graph_builder = StateGraph(State)
    
    # Add all agent nodes
    graph_builder.add_node("supervisor", partial(supervisor, workflow=workflow))
    graph_builder.add_node("query_enhancement", partial(query_enhancement_agent, workflow=workflow))
    graph_builder.add_node("tool_execution", partial(tool_execution_agent, workflow=workflow))
    graph_builder.add_node("response_generation", partial(response_generation_agent, workflow=workflow))
    graph_builder.add_node("quality_assurance", partial(quality_assurance_agent, workflow=workflow))
    
    # Set entry point
    graph_builder.set_entry_point("supervisor")
    
    # Add conditional routing from supervisor
    graph_builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "query_enhancement": "query_enhancement",
            "tool_execution": "tool_execution",
            "response_generation": "response_generation",
            "quality_assurance": "quality_assurance",
            "end": END
        }
    )
    
    # Each agent returns to supervisor for next decision
    graph_builder.add_edge("query_enhancement", "supervisor")
    graph_builder.add_edge("tool_execution", "supervisor")
    graph_builder.add_edge("response_generation", "supervisor")
    graph_builder.add_edge("quality_assurance", "supervisor")
    
    return graph_builder.compile(checkpointer=memory)


# --------------------------
# Usage Example
# --------------------------
async def run_agent(query: str, retriever, thread_id: str = "default"):
    """Helper function to run the agent"""
    workflow = Workflow(retriever, llm, llm_router)
    graph = build_graph(workflow)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "iteration_count": 0
    }
    
    final_state = await graph.ainvoke(initial_state, config)
    
    # Return final response
    return final_state["messages"][-1].content