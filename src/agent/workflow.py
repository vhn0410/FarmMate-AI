from src.tools.knowledge_base_tool import KnowledgeBaseService, build_kb_tool
from src.tools.sensorthings_tool import sensorthings_search
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Literal
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
    sensor_data: dict  # Store parsed sensor data
    kb_context: str    # Store KB context
    query_type: str    # Track query type: 'sensor_check' or 'knowledge'

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
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --------------------------
# Graph nodes
# --------------------------
async def planner(state: State, workflow: Workflow):
    """Planner decides which tools to call based on query type."""
    system = SystemMessage(content=(
        "Bạn là agent phân loại câu hỏi của nông dân.\n\n"
        "PHÂN LOẠI CÂU HỎI:\n\n"
        "🔍 NHÓM 1 - KIỂM TRA/THEO DÕI DỮ LIỆU THỰC TẾ (sensor_check):\n"
        "Các từ khóa: kiểm tra, theo dõi, hiện tại, bây giờ, đang, giờ này\n"
        "Ví dụ:\n"
        "✓ 'Kiểm tra đất của tôi'\n"
        "✓ 'Nhiệt độ hiện tại như thế nào?'\n"
        "✓ 'Độ ẩm bây giờ ra sao?'\n"
        "✓ 'Xem dữ liệu cảm biến'\n"
        "→ GỌI: sensorthings_search + search_knowledge_base\n\n"
        
        "📚 NHÓM 2 - KIẾN THỨC/HƯỚNG DẪN (knowledge):\n"
        "Các từ khóa: là gì, cách, kỹ thuật, hướng dẫn, phương pháp, giai đoạn, bệnh, sâu\n"
        "Ví dụ:\n"
        "✓ 'Cuối giai đoạn sinh trưởng là gì?'\n"
        "✓ 'Hướng dẫn bón phân'\n"
        "✓ 'Cách trị bệnh đạo ôn'\n"
        "✓ 'Kỹ thuật làm đất'\n"
        "→ GỌI: search_knowledge_base (CHỈ MỘT TOOL)\n\n"
        
        "QUAN TRỌNG:\n"
        "- Nếu câu hỏi KHÔNG có từ 'kiểm tra/theo dõi/hiện tại/bây giờ' → CHỈ gọi search_knowledge_base\n"
        "- Nếu câu hỏi về ĐỊNH NGHĨA, GIẢI THÍCH, HƯỚNG DẪN → CHỈ gọi search_knowledge_base\n"
        "- LUÔN gọi ít nhất 1 tool\n"
    ))

    messages = [system] + state["messages"]
    result = await workflow.llm_with_tools.ainvoke(messages)
    return {"messages": [result]}

def tools_router(state: State) -> Literal["tool_node", "responder"]:
    """Route to tools or responder based on last message"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    return "responder"

async def tool_node(state: State, workflow: Workflow):
    """Execute tool calls and store structured data"""
    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    tool_messages = []
    
    # RESET sensor_data when starting new tool execution
    sensor_data = {}
    kb_context = state.get("kb_context", "")
    query_type = "knowledge"  # Default
    
    has_sensor_call = any(call["name"] == "sensorthings_search" for call in tool_calls)
    if has_sensor_call:
        query_type = "sensor_check"

    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        tool_id = call.get("id")
        
        try:
            if name == "search_knowledge_base":
                q = args.get("query") if isinstance(args, dict) else args
                out = await workflow.search_knowledge_base.ainvoke(q)
                
                try:
                    parsed = json.loads(out)
                    if parsed.get("type") == "docs":
                        text = f"[Kiến thức từ KB về: {parsed.get('query')}]\n"
                        for r in parsed.get("results", [])[:5]:
                            text += f"• {r.get('excerpt')}\n"
                        kb_context += text + "\n"
                    else:
                        text = parsed.get("text", str(parsed))
                        kb_context += text + "\n"
                except Exception:
                    text = out
                    kb_context += text + "\n"
                    
                tool_messages.append(ToolMessage(content=text, tool_call_id=tool_id))
                
            elif name == "sensorthings_search":
                q = args.get("query") if isinstance(args, dict) else args
                out = await sensorthings_search.ainvoke(q)
                
                if isinstance(out, dict):
                    # Store structured sensor data for analysis
                    sensor_data.update(out)
                    
                    pretty = json.dumps(out, ensure_ascii=False, indent=2)
                    if len(pretty) > 4000:
                        pretty = pretty[:4000] + "... [truncated]"
                    tool_messages.append(ToolMessage(content=pretty, tool_call_id=tool_id))
                else:
                    tool_messages.append(ToolMessage(content=str(out), tool_call_id=tool_id))
            else:
                tool_messages.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tool_id))
                
        except Exception as e:
            tool_messages.append(ToolMessage(content=f"Tool {name} error: {str(e)}", tool_call_id=tool_id))

    return {
        "messages": tool_messages,
        "sensor_data": sensor_data,
        "kb_context": kb_context,
        "query_type": query_type
    }

async def responder(state: State, workflow: Workflow):
    """Generate intelligent analysis and recommendations"""
    
    # Check query type and sensor data
    query_type = state.get("query_type", "knowledge")
    has_sensor_data = bool(state.get("sensor_data"))
    
    # Only use sensor analysis system prompt if BOTH conditions are true
    if query_type == "sensor_check" and has_sensor_data:
        system = SystemMessage(content=(
            "Bạn là chuyên gia nông nghiệp AI với khả năng PHÂN TÍCH CHUYÊN SÂU.\n\n"
            "NHIỆM VỤ CỦA BẠN:\n"
            "1. PHÂN TÍCH DỮ LIỆU:\n"
            "   - So sánh từng chỉ số với ngưỡng chuẩn từ knowledge base\n"
            "   - Xác định chỉ số NÀO TỐT, NÀO CẦN CẢI THIỆN, NÀO NGUY HIỂM\n"
            "   - Giải thích TẠI SAO (ví dụ: N thấp → thiếu đạm → cây vàng lá)\n"
            "\n"
            "2. CẢNH BÁO RÕ RÀNG:\n"
            "   - ⚠️ CẢNH BÁO: nếu có chỉ số nguy hiểm\n"
            "   - ⚡ KHẨN CẤP: nếu cần xử lý ngay\n"
            "   - ✅ BÌNH THƯỜNG: nếu mọi thứ ổn\n"
            "\n"
            "3. KHUYẾN NGHỊ CỤ THỂ:\n"
            "   - Tên phân bón/thuốc cần dùng (VD: Urê, NPK 16-16-8, DAP)\n"
            "   - Liều lượng chính xác (VD: 50kg Urê/ha)\n"
            "   - Thời điểm bón (VD: 7-10 ngày sau cấy)\n"
            "   - Cách bón (VD: rải đều, hòa nước)\n"
            "\n"
            "4. KẾ HOẠCH THEO DÕI:\n"
            "   - Kiểm tra lại sau bao lâu?\n"
            "   - Cần quan sát triệu chứng gì?\n"
            "\n"
            "5. ĐỊNH DẠNG:\n"
            "   - Dùng emoji để dễ nhìn (⚠️✅📊💡🌾)\n"
            "   - Ngắn gọn, súc tích, dễ hiểu\n"
            "   - Ưu tiên hành động TỨC THỜI\n"
            "\n"
            "LƯU Ý:\n"
            "- KHÔNG chỉ liệt kê số liệu\n"
            "- KHÔNG nói chung chung 'cần theo dõi'\n"
            "- PHẢI đưa ra HÀNH ĐỘNG CỤ THỂ\n"
            "- Trả lời bằng TIẾNG VIỆT\n"
        ))
    else:
        # Knowledge-only system prompt
        system = SystemMessage(content=(
            "Bạn là chuyên gia nông nghiệp Việt Nam.\n\n"
            "NHIỆM VỤ:\n"
            "- Trả lời dựa HOÀN TOÀN trên kiến thức từ knowledge base\n"
            "- KHÔNG đề cập đến dữ liệu cảm biến nếu câu hỏi không yêu cầu\n"
            "- Giải thích RÕ RÀNG, DỄ HIỂU các khái niệm\n"
            "- Đưa ra SỐ LIỆU CỤ THỂ nếu có (liều lượng, thời gian)\n"
            "- Chia thành CÁC BƯỚC nếu là hướng dẫn\n"
            "\n"
            "ĐỊNH DẠNG:\n"
            "- Dùng emoji phù hợp để dễ đọc\n"
            "- Ngắn gọn, tập trung vào câu hỏi\n"
            "- TIẾNG VIỆT dễ hiểu\n"
            "\n"
            "LƯU Ý:\n"
            "- Nếu KB không có thông tin → nói rõ 'Tôi không tìm thấy thông tin này trong tài liệu'\n"
            "- KHÔNG bịa đặt thông tin không có trong KB\n"
        ))

    messages = [system] + state["messages"]
    print("Responder messages:", messages)
    try:
        
        # 1. Gọi LLM an toàn
        result = await workflow.llm.ainvoke(messages, stream=False)

        #  2. Extract content thay vì trả nguyên object
        text = result.content if hasattr(result, "content") else str(result)

        #  3. Trả về JSON serializable object
        return {"messages": [AIMessage(content=text)]}
    except asyncio.CancelledError:
        # Khi bị hủy bởi LangGraph/LangSmith, ta không cần xem là lỗi
        print("Responder task was cancelled (likely by LangGraph).")
        return None

# --------------------------
# Build graph
# --------------------------
def build_graph(workflow: Workflow):
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("planner", partial(planner, workflow=workflow))
    graph_builder.add_node("tool_node", partial(tool_node, workflow=workflow))
    graph_builder.add_node("responder", partial(responder, workflow=workflow))
    
    # Set entry point
    graph_builder.set_entry_point("planner")
    
    # Add edges
    graph_builder.add_conditional_edges(
        "planner",
        tools_router,
        {
            "tool_node": "tool_node",
            "responder": "responder"
        }
    )
    graph_builder.add_edge("tool_node", "responder")
    graph_builder.add_edge("responder", END)
    
    return graph_builder.compile(checkpointer=memory)