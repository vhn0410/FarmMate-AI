
from src.tools.knowledge_base_tool import KnowledgeBaseService, build_kb_tool
from src.tools.sensorthings_tool import sensorthings_search
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from dotenv import load_dotenv
import json
load_dotenv()



# --------------------------
# Memory & Graph State
# --------------------------
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Workflow:
    retriever = None  # inject từ ngoài
    def __init__(self, retriever):
        self.retriever = retriever
   

# --------------------------
# LLM setup
# --------------------------
# Choose whichever LLM is available in your env. We'll make two llms: one for planning/router (no tools) and one bound-to-tools.
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # optional smaller router

# Bind tools for the model that will do generation with tool-calls
knowledge_base = KnowledgeBaseService(Workflow.retriever)
search_knowledge_base = build_kb_tool(knowledge_base)
llm_with_tools = llm_router.bind_tools(tools=[search_knowledge_base, sensorthings_search])

# --------------------------
# Nodes: planner, tool_node, responder
# --------------------------
async def planner(state: State):
    """Planner decides whether to call tools and with what queries. 
    It should NOT produce final user-facing answer - only tool calls.
    Output: appended AI planning message (may include tool_calls).
    """
    # Build a short system prompt to the model
    system = SystemMessage(content=(
        "You are a routing agent that analyzes user queries and decides which tools to call.\n\n"
        "Rules:\n"
        "- If user asks about devices, sensors, datastreams, measurements, or current values → call sensorthings_search\n"
        "- If user asks about agronomy, treatments, recommendations, or farming knowledge → call search_knowledge_base\n"
        "- You MUST call at least one appropriate tool based on the query\n"
        "- Output ONLY tool calls - do NOT generate conversational responses or explanations\n"
        "- Do NOT produce the final answer - that's the responder's job\n"
    ))

    # Send planner message to llm_with_tools so it can emit tool_calls
    messages = [system] + state["messages"]
    result = await llm_with_tools.ainvoke(messages)
    
    # append (preserve history)
    return {"messages": state["messages"] + [result]}


async def tools_router(state: State):
    """Route based on whether planner produced tool calls."""
    last_message = state["messages"][-1]
    
    # Check if this is an AI message with tool calls
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        # If planner didn't call tools, go straight to responder
        return "responder"


async def tool_node(state: State):
    """Execute tools and return results as ToolMessages."""
    tool_calls = getattr(state["messages"][-1], "tool_calls", [])
    tool_messages = []
    
    for call in tool_calls:
        name = call["name"]
        args = call["args"]
        tool_id = call.get("id")
        
        try:
            if name == "search_knowledge_base":
                q = args.get("query") if isinstance(args, dict) else args
                out = await search_knowledge_base.ainvoke(q)
                # ensure JSON
                try:
                    parsed = json.loads(out)
                    # create a textual summary for the LLM
                    if parsed.get("type") == "docs":
                        text = f"[KB Results for: {parsed.get('query')}]\n"
                        for r in parsed.get("results", []):
                            text += f"- {r.get('title')} | {r.get('source')} | {r.get('excerpt')}\n"
                    else:
                        text = parsed.get("text") if parsed.get("text") else str(parsed)
                except Exception:
                    text = out
                tool_messages.append(ToolMessage(content=text, tool_call_id=tool_id))
                
            elif name == "sensorthings_search":
                q = args.get("query") if isinstance(args, dict) else args
                out = await sensorthings_search.ainvoke(q)
                # out is dict
                if isinstance(out, dict):
                    # pretty but succinct
                    pretty = json.dumps(out, ensure_ascii=False, indent=2)
                    # limit size
                    if len(pretty) > 6000:
                        pretty = pretty[:6000] + "... [truncated]"
                    tool_messages.append(ToolMessage(content=pretty, tool_call_id=tool_id))
                else:
                    tool_messages.append(ToolMessage(content=str(out), tool_call_id=tool_id))
                    
            else:
                tool_messages.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tool_id))
                
        except Exception as e:
            tool_messages.append(ToolMessage(content=f"Tool {name} error: {str(e)}", tool_call_id=tool_id))
            
    # Return appended messages (so history preserved)
    return {"messages": state["messages"] + tool_messages}


async def responder(state: State):
    """Responder: use the latest context (including tool messages) to produce final answer to user.
    This is the ONLY node that should generate the final user-facing response.
    Append the response and preserve history.
    """
    # System prompt to instruct the model to use the documents/tools
    system = SystemMessage(content=(
        "Bạn là trợ lý nông nghiệp chuyên nghiệp của Việt Nam. "
        "Dựa trên thông tin từ các công cụ (tool outputs) và knowledge base, "
        "hãy trả lời câu hỏi của nông dân một cách ngắn gọn, rõ ràng và hữu ích.\n\n"
        "Quy tắc quan trọng:\n"
        "- Trả lời bằng TIẾNG VIỆT\n"
        "- Nếu có giá trị cảm biến bất thường (được đánh dấu _validator), hãy cảnh báo và đề xuất kiểm tra lại\n"
        "- Đưa ra số liệu cụ thể khi có thể\n"
        "- Đưa ra các bước hành động thiết thực, dễ thực hiện\n"
        "- Nếu thông tin không đủ, hãy nói rõ và hỏi thêm thông tin cần thiết\n"
        "- QUAN TRỌNG: Chỉ trả lời MỘT LẦN duy nhất, không lặp lại nội dung\n"
    ))
    
    messages = [system] + state["messages"]
    
    # Use llm (without tool-binding) to produce final answer
    result = await llm.ainvoke(messages)
    
    return {"messages": state["messages"] + [result]}

# --------------------------
# Graph assembly
# --------------------------
graph_builder = StateGraph(State)
graph_builder.add_node("planner", planner)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("responder", responder)

# Set entry point
graph_builder.set_entry_point("planner")

# Conditional routing from planner
graph_builder.add_conditional_edges("planner", tools_router)

# Tool node always goes to responder
graph_builder.add_edge("tool_node", "responder")

# Responder goes to END (no loop back)
graph_builder.add_edge("responder", END)

graph = graph_builder.compile(checkpointer=memory)