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
# Graph nodes
# --------------------------
async def planner(state: State, workflow: Workflow):
    """Planner decides which tools to call based on query type."""
    system = SystemMessage(content=(
        "Bạn là agent phân tích câu hỏi của nông dân và quyết định công cụ nào cần dùng.\n\n"
        "LUẬT QUAN TRỌNG:\n"
        "1. Nếu câu hỏi về KIỂM TRA/THEO DÕI dữ liệu thực tế (nhiệt độ, độ ẩm, NPK, pH, cảm biến):\n"
        "   → GỌI sensorthings_search TRƯỚC\n"
        "   → SAU ĐÓ GỌI search_knowledge_base để lấy ngưỡng chuẩn và khuyến nghị\n"
        "\n"
        "2. Nếu câu hỏi về KIẾN THỨC/HƯỚNG DẪN chung (kỹ thuật, phương pháp, bệnh hại):\n"
        "   → CHỈ GỌI search_knowledge_base\n"
        "\n"
        "3. LUÔN GỌI ÍT NHẤT 1 CÔNG CỤ. Không trả lời trực tiếp.\n"
        "\n"
        "Ví dụ:\n"
        "- 'Kiểm tra đất' → sensorthings_search + search_knowledge_base\n"
        "- 'Hướng dẫn bón phân' → search_knowledge_base\n"
        "- 'Nhiệt độ hiện tại' → sensorthings_search + search_knowledge_base\n"
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
    sensor_data = state.get("sensor_data", {})
    kb_context = state.get("kb_context", "")

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
        "kb_context": kb_context
    }

async def responder(state: State, workflow: Workflow):
    """Generate intelligent analysis and recommendations"""
    
    # Check if we have sensor data
    has_sensor_data = bool(state.get("sensor_data"))
    
    if has_sensor_data:
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
            "   - ✅ BÌN THƯỜNG: nếu mọi thứ ổn\n"
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
        system = SystemMessage(content=(
            "Bạn là chuyên gia nông nghiệp Việt Nam.\n"
            "Dựa trên kiến thức từ knowledge base, hãy:\n"
            "1. Trả lời NGẮN GỌN, THỰC TẾ\n"
            "2. Đưa ra SỐ LIỆU CỤ THỂ (liều lượng, thời gian)\n"
            "3. Chia thành CÁC BƯỚC RÕ RÀNG\n"
            "4. Dùng TIẾNG VIỆT dễ hiểu\n"
            "5. Thêm emoji để dễ đọc nếu phù hợp\n"
        ))

    messages = [system] + state["messages"]
    result = await workflow.llm.ainvoke(messages)
    return {"messages": [result]}

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