# from src.tools.knowledge_base_tool import KnowledgeBaseService, build_kb_tool
# from src.tools.sensorthings_tool import sensorthings_search
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from typing import TypedDict, Annotated, Literal, Optional
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# import json
# from functools import partial
# import asyncio
# load_dotenv()

# # --------------------------
# # Memory & Graph State
# # --------------------------
# memory = MemorySaver()

# def add_messages(left, right):
#     """Custom reducer to prevent message duplication"""
#     if not isinstance(left, list):
#         left = [left]
#     if not isinstance(right, list):
#         right = [right]
#     return left + right

# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     original_query: str  # Câu hỏi gốc
#     enhanced_query: str  # Câu hỏi đã enhance
#     tool_plan: dict  # Kế hoạch gọi tools
#     sensor_data: dict  # Dữ liệu cảm biến
#     kb_context: str  # Context từ KB
#     draft_response: str  # Câu trả lời nháp
#     qa_feedback: dict  # Feedback từ QA agent
#     iteration_count: int  # Số lần iteration
#     next_agent: str  # Agent tiếp theo
#     needs_improvement: bool  # Flag để quyết định có cần cải thiện không
#     has_kb_data: bool  # Flag kiểm tra có dữ liệu KB không
#     has_sensor_data: bool  # Flag kiểm tra có dữ liệu sensor không
#     is_small_talk: bool  # Flag kiểm tra small talk

# SUPERVISOR = "supervisor"
# QUERY_ENHANCER = "query_enhancement"
# TOOL_EXECUTOR = "tool_execution"
# RESPONSE_GENERATOR = "response_generation"
# QUALITY_ASSURER = "quality_assurance"


# # --------------------------
# # Workflow class
# # --------------------------
# class Workflow:
#     def __init__(self, retriever, llm, llm_router):
#         self.retriever = retriever
#         self.llm = llm
#         self.llm_router = llm_router
        
#         # Bind tools
#         self.knowledge_base = KnowledgeBaseService(self.retriever)
#         self.search_knowledge_base = build_kb_tool(self.knowledge_base)
#         self.llm_with_tools = self.llm_router.bind_tools(
#             tools=[self.search_knowledge_base, sensorthings_search]
#         )

# # --------------------------
# # LLM setup
# # --------------------------
# llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
# llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# # --------------------------
# # Agent Nodes
# # --------------------------

# async def supervisor(state: State, workflow: Workflow):
#     """
#     Supervisor Agent - Điều phối workflow
#     Quyết định agent nào chạy tiếp theo
#     """
#     print("\n[Supervisor] Current State:", state)

#     iteration = state.get("iteration_count", 0)
#     last_agent = state.get("next_agent", "")

#     # Lấy intent safe
#     tool_plan = state.get("tool_plan", {})
    
#     if tool_plan != {}:
#         intent = tool_plan.get("intent", "")
#     else:
#         intent = ""

#     # ✅ FIX 1: Nếu người dùng chỉ chat xã giao, skip tool execution và response generation
#     if last_agent == "query_enhancement" and intent == "small_talk":
#         return {"next_agent": "small_talk_response"}

#     if iteration == 0:
#         return {"next_agent": "query_enhancement", "iteration_count": 1}

#     if last_agent == "query_enhancement":
#         return {"next_agent": "tool_execution"}

#     elif last_agent == "tool_execution":
#         return {"next_agent": "response_generation"}

#     elif last_agent == "response_generation":
#         if not state.get("qa_feedback"):
#             return {"next_agent": "quality_assurance"}
#         else:
#             return {"next_agent": "end"}

#     elif last_agent == "quality_assurance":
#         needs_improvement = state.get("needs_improvement", False)
#         if needs_improvement and iteration < 3:
#             return {"next_agent": "response_generation", "iteration_count": iteration + 1}
#         else:
#             return {"next_agent": "end"}
    
#     elif last_agent == "small_talk_response":
#         return {"next_agent": "end"}

#     return {"next_agent": "end"}


# async def query_enhancement_agent(state: State, workflow: Workflow):
#     """
#     Query Enhancement Agent - IMPROVED VERSION
#     - Phân tích ý định người dùng bằng LLM
#     - Mở rộng câu hỏi với context nông nghiệp
#     - Xác định loại thông tin cần thiết
#     """
#     print("\n[Query Enhancement] Original State:", state)

#     system = SystemMessage(content=(
#         "Bạn là chuyên gia phân tích ý định người dùng trong lĩnh vực nông nghiệp thông minh.\n\n"
        
#         "# NHIỆM VỤ:\n"
#         "Phân tích câu hỏi và xác định:\n"
#         "1. Người dùng muốn gì? (intent)\n"
#         "2. Câu hỏi đã đủ rõ chưa? Cần mở rộng thêm gì?\n"
#         "3. Cần dữ liệu gì để trả lời? (sensor data, knowledge base, hay không cần gì)\n\n"
        
#         "# PHÂN LOẠI INTENT:\n\n"
        
#         "## 🗣️ small_talk\n"
#         "Câu hỏi KHÔNG liên quan đến nông nghiệp:\n"
#         "- Chào hỏi xã giao: 'xin chào', 'hello', 'chào bạn'\n"
#         "- Giới thiệu bản thân: 'tôi là...', 'mình tên...', 'tên tôi'\n"
#         "- Câu hỏi về danh tính: 'bạn là ai', 'tên bạn', 'tôi là ai'\n"
#         "- Câu hỏi cá nhân: 'tôi đẹp không', 'hôm nay thứ mấy'\n"
#         "- Cảm ơn/tạm biệt: 'thanks', 'cảm ơn', 'bye'\n"
#         "→ data_needs: [] (KHÔNG cần gọi tool)\n\n"
        
#         "## 📊 check_sensor_data\n"
#         "Người dùng muốn KIỂM TRA/XEM dữ liệu thực tế:\n"
#         "- Chỉ số cảm biến: 'nhiệt độ bao nhiêu', 'độ ẩm hiện tại', 'pH đất'\n"
#         "- Trạng thái thiết bị: 'cảm biến đất số 1', 'thiết bị hoạt động'\n"
#         "- So sánh xu hướng: 'nhiệt độ có tăng không', 'NPK thay đổi'\n"
#         "→ data_needs: ['sensor_data', 'knowledge_base']\n\n"
        
#         "## 📚 ask_knowledge\n"
#         "Người dùng muốn HỌC/HIỂU kiến thức:\n"
#         "- Kiến thức chung: 'cách trồng lúa', 'chu kỳ sinh trưởng'\n"
#         "- Giải thích: 'tại sao lá vàng', 'nguyên nhân cây chết'\n"
#         "- Hướng dẫn: 'cách bón phân', 'phòng trừ sâu bệnh'\n"
#         "→ data_needs: ['knowledge_base']\n\n"
        
#         "## 💡 consultation\n"
#         "Người dùng muốn TƯ VẤN dựa trên tình huống:\n"
#         "- Giải quyết vấn đề: 'đất tôi bị chua, làm sao'\n"
#         "- Đề xuất hành động: 'nên bón phân gì', 'khi nào thu hoạch'\n"
#         "- Tối ưu hóa: 'tăng năng suất', 'giảm chi phí'\n"
#         "→ data_needs: ['sensor_data', 'knowledge_base']\n\n"
        
#         "# LUẬT QUAN TRỌNG:\n"
#         "1. Nếu KHÔNG liên quan nông nghiệp → intent = 'small_talk', data_needs = []\n"
#         "2. Nếu hỏi SỐ LIỆU/TRẠNG THÁI → bắt buộc có 'sensor_data'\n"
#         "3. Nếu chỉ hỏi KIẾN THỨC → chỉ cần 'knowledge_base'\n"
#         "4. Nếu TƯ VẤN → cần cả hai\n"
#         "5. MỞ RỘNG câu hỏi với thuật ngữ chuyên môn\n\n"
        
#         "# VÍ DỤ:\n\n"
#         "Input: 'xin chào'\n"
#         "Output: {\"intent\": \"small_talk\", \"enhanced_query\": \"xin chào\", \"data_needs\": [], \"context\": \"Chào hỏi xã giao\"}\n\n"
        
#         "Input: 'tôi là Vũ'\n"
#         "Output: {\"intent\": \"small_talk\", \"enhanced_query\": \"tôi là Vũ\", \"data_needs\": [], \"context\": \"Giới thiệu bản thân\"}\n\n"
        
#         "Input: 'tôi là ai'\n"
#         "Output: {\"intent\": \"small_talk\", \"enhanced_query\": \"tôi là ai\", \"data_needs\": [], \"context\": \"Câu hỏi về danh tính\"}\n\n"
        
#         "Input: 'kiểm tra đất'\n"
#         "Output: {\"intent\": \"check_sensor_data\", \"enhanced_query\": \"Kiểm tra các chỉ số đất: NPK (Nitơ, Lân, Kali), pH, độ ẩm đất, nhiệt độ đất\", \"data_needs\": [\"sensor_data\", \"knowledge_base\"], \"context\": \"Kiểm tra tình trạng đất\"}\n\n"
        
#         "Input: 'cách bón phân cho lúa'\n"
#         "Output: {\"intent\": \"ask_knowledge\", \"enhanced_query\": \"Hướng dẫn bón phân cho lúa: loại phân, liều lượng, thời điểm, cách bón\", \"data_needs\": [\"knowledge_base\"], \"context\": \"Hỏi kỹ thuật bón phân\"}\n\n"
        
#         "# OUTPUT FORMAT:\n"
#         "Trả về JSON hợp lệ (KHÔNG dùng markdown ```json):\n"
#         "{\"intent\": \"...\", \"enhanced_query\": \"...\", \"data_needs\": [...], \"context\": \"...\"}\n\n"
        
#         "Bây giờ hãy phân tích câu hỏi sau:"
#     ))
    
#     # print("\n[Query Enhancement] Original Query:", state)

#     original_query = state["messages"][-1].content if state["messages"] else state.get("original_query", "")
#     # print(f"[Query Enhancement] Analyzing query: {original_query}")

#     messages = [system, HumanMessage(content=original_query)]
#     result = await workflow.llm_router.ainvoke(messages)
    
#     # ✅ FIX 2: Dùng LLM để phân tích thay vì hardcode
#     try:
#         content = result.content.strip()
#         # Remove markdown code blocks if present
#         if content.startswith("```"):
#             content = content.split("```")[1]
#             if content.startswith("json"):
#                 content = content[4:]
#         enhanced = json.loads(content.strip())
#     except Exception as e:
#         # print(f"[Query Enhancement] Parse error: {e}, using default")
#         # Fallback: check if it's small talk manually
#         if any(pattern in original_query.lower() for pattern in [
#             "xin chào", "hello", "hi", "chào",
#             "tôi là", "mình là", "tên tôi", "tên mình",
#             "bạn là ai", "tôi là ai", "ai đó",
#             "cảm ơn", "thank", "bye", "tạm biệt"
#         ]):
#             enhanced = {
#                 "intent": "small_talk",
#                 "enhanced_query": original_query,
#                 "data_needs": [],
#                 "context": "Câu hỏi không liên quan nông nghiệp"
#             }
#         else:
#             enhanced = {
#                 "intent": "ask_knowledge",
#                 "enhanced_query": original_query,
#                 "data_needs": ["knowledge_base"],
#                 "context": ""
#             }
    
#     # print(f"[Query Enhancement] Intent: {enhanced.get('intent')}, Data needs: {enhanced.get('data_needs')}")
    
#     return {
#         "original_query": original_query,
#         "enhanced_query": enhanced.get("enhanced_query", original_query),
#         "tool_plan": enhanced,
#         "messages": [AIMessage(content=f"[Enhanced Query: {enhanced.get('enhanced_query')}]")]
#     }


# async def tool_execution_agent(state: State, workflow: Workflow):
#     """
#     Tool Execution Agent - Gọi tools dựa trên tool_plan
#     - Thu thập dữ liệu từ sensors và knowledge base
#     """
#     print("\n[Tool Execution] Current State:", state)
#     tool_plan = state.get("tool_plan", {})
#     data_needs = tool_plan.get("data_needs", [])
#     enhanced_query = state.get("enhanced_query", state.get("original_query", ""))
    
#     # ✅ FIX 3: Nếu intent là small_talk, KHÔNG gọi tools
#     if tool_plan.get("intent") == "small_talk" or not data_needs:
#         # print(f"[Tool Execution] Skipping tools for small_talk")
#         return {
#             "sensor_data": {},
#             "kb_context": "",
#             "has_kb_data": False,
#             "has_sensor_data": False,
#             "messages": [ToolMessage(content="Skipped tools for small talk", tool_call_id="none")]
#         }
    
#     sensor_data = {}
#     kb_context = ""
#     tool_messages = []
#     has_kb_data = False
#     has_sensor_data = False
    
#     # print(f"\n[Tool Execution] Query: {enhanced_query}")
#     # print(f"[Tool Execution] Data needs: {data_needs}")
    
#     # 1. GỌI SENSOR TOOL (nếu cần)
#     if "sensor_data" in data_needs:
#         print(f"[Tool Execution] Calling sensorthings_search...")
#         try:
#             sensor_result = await sensorthings_search.ainvoke(enhanced_query)
#             print(f"[Tool Execution] Sensor result type: {type(sensor_result)}")
            
#             if isinstance(sensor_result, dict) and sensor_result:
#                 sensor_data.update(sensor_result)
#                 has_sensor_data = True
#                 tool_messages.append(
#                     ToolMessage(
#                         content=json.dumps(sensor_result, ensure_ascii=False, indent=2),
#                         tool_call_id="sensor_tool"
#                     )
#                 )
#                 print(f"[Tool Execution] ✅ Got sensor data: {len(sensor_data)} keys")
#             else:
#                 tool_messages.append(
#                     ToolMessage(content="Không có dữ liệu cảm biến", tool_call_id="sensor_tool")
#                 )
#                 print(f"[Tool Execution] ⚠️ No sensor data")
#         except Exception as e:
#             error_msg = f"Lỗi khi lấy dữ liệu cảm biến: {str(e)}"
#             tool_messages.append(
#                 ToolMessage(content=error_msg, tool_call_id="sensor_tool")
#             )
#             print(f"[Tool Execution] ❌ Sensor error: {e}")
    
#     # 2. GỌI KNOWLEDGE BASE TOOL (nếu cần)
#     if "knowledge_base" in data_needs:
#         print(f"[Tool Execution] Calling search_knowledge_base...")
#         try:
#             kb_result = await workflow.search_knowledge_base.ainvoke(enhanced_query)
#             print(f"[Tool Execution] KB result length: {len(str(kb_result))}")
            
#             if kb_result and len(str(kb_result).strip()) > 10:
#                 kb_context = kb_result
#                 has_kb_data = True
#                 tool_messages.append(
#                     ToolMessage(content=kb_result, tool_call_id="kb_tool")
#                 )
#                 print(f"[Tool Execution] ✅ Got KB data")
#             else:
#                 kb_context = "Không tìm thấy thông tin liên quan trong cơ sở kiến thức."
#                 tool_messages.append(
#                     ToolMessage(content=kb_context, tool_call_id="kb_tool")
#                 )
#                 print(f"[Tool Execution] ⚠️ No KB data")
#         except Exception as e:
#             error_msg = f"Lỗi khi tìm kiếm knowledge base: {str(e)}"
#             kb_context = error_msg
#             tool_messages.append(
#                 ToolMessage(content=error_msg, tool_call_id="kb_tool")
#             )
#             print(f"[Tool Execution] ❌ KB error: {e}")
    
#     return {
#         "sensor_data": sensor_data,
#         "kb_context": kb_context,
#         "has_kb_data": has_kb_data,
#         "has_sensor_data": has_sensor_data,
#         "messages": tool_messages
#     }


# async def small_talk_response_agent(state: State, workflow: Workflow):
#     """
#     Agent chuyên xử lý small talk
#     Trả lời thân thiện cho các câu hỏi không liên quan nông nghiệp
#     """
#     print("\n[Small Talk Response] Current State:", state)
#     original_query = state.get("original_query", "").lower()
    
#     # Phân loại loại small talk
#     if any(greet in original_query for greet in ["xin chào", "hello", "hi", "chào"]):
#         response = (
#             "Xin chào! 👋 Tôi là trợ lý nông nghiệp thông minh. "
#             "Tôi có thể giúp bạn:\n\n"
#             "📊 Kiểm tra dữ liệu cảm biến (nhiệt độ, độ ẩm, NPK, pH)\n"
#             "📚 Tư vấn kỹ thuật canh tác\n"
#             "💡 Giải đáp thắc mắc về cây trồng\n\n"
#             "Bạn cần hỗ trợ gì không?"
#         )
    
#     elif any(intro in original_query for intro in ["tôi là", "mình là", "tên tôi", "tên mình"]):
#         # Trích xuất tên từ câu (nếu có)
#         name = ""
#         if "tôi là" in original_query:
#             name = original_query.split("tôi là")[-1].strip()
#         elif "mình là" in original_query:
#             name = original_query.split("mình là")[-1].strip()
        
#         if name:
#             response = (
#                 f"Chào {name.title()}! 🌾 Rất vui được làm quen.\n\n"
#                 f"Tôi là trợ lý nông nghiệp, có thể giúp bạn:\n"
#                 f"- Kiểm tra dữ liệu đất và cây trồng\n"
#                 f"- Tư vấn bón phân, chăm sóc cây\n"
#                 f"- Giải đáp về sâu bệnh\n\n"
#                 f"Bạn đang trồng gì và cần hỗ trợ gì không?"
#             )
#         else:
#             response = (
#                 "Chào bạn! Rất vui được làm quen. "
#                 "Tôi có thể giúp bạn với các vấn đề về nông nghiệp. "
#                 "Bạn cần hỗ trợ gì không?"
#             )
    
#     elif any(who in original_query for who in ["bạn là ai", "tên bạn", "ai đó", "tôi là ai"]):
#         if "tôi là ai" in original_query:
#             response = (
#                 "Tôi là trợ lý AI nông nghiệp, tôi không có thông tin về danh tính của bạn. 🤔\n\n"
#                 "Nhưng tôi có thể giúp bạn với:\n"
#                 "- Phân tích đất\n"
#                 "- Tư vấn cây trồng\n"
#                 "- Giải đáp kỹ thuật\n\n"
#                 "Bạn có câu hỏi gì về nông nghiệp không?"
#             )
#         else:
#             response = (
#                 "Tôi là trợ lý nông nghiệp thông minh! 🤖🌾\n\n"
#                 "Tôi được tạo ra để hỗ trợ nông dân với:\n"
#                 "✅ Dữ liệu từ cảm biến IoT\n"
#                 "✅ Kiến thức canh tác hiện đại\n"
#                 "✅ Tư vấn chuyên sâu\n\n"
#                 "Bạn cần giúp gì về nông nghiệp không?"
#             )
    
#     elif any(thanks in original_query for thanks in ["cảm ơn", "thanks", "thank you"]):
#         response = (
#             "Không có gì! 😊 Rất vui được hỗ trợ bạn.\n\n"
#             "Nếu có thêm câu hỏi về nông nghiệp, cứ hỏi tôi bất cứ lúc nào nhé! 🌱"
#         )
    
#     elif any(bye in original_query for bye in ["bye", "tạm biệt", "hẹn gặp lại"]):
#         response = (
#             "Tạm biệt! 👋 Chúc bạn mùa màng bội thu! 🌾\n\n"
#             "Hẹn gặp lại bạn sớm!"
#         )
    
#     else:
#         # Default small talk response
#         response = (
#             "Tôi là trợ lý nông nghiệp. Tôi có thể giúp bạn với các câu hỏi về:\n"
#             "- Kiểm tra dữ liệu cảm biến\n"
#             "- Kỹ thuật canh tác\n"
#             "- Chăm sóc cây trồng\n\n"
#             "Bạn có câu hỏi gì không?"
#         )
    
#     return {
#         "draft_response": response,
#         # "messages": [AIMessage(content=response)],
#         "is_small_talk": True
#     }


# async def response_generation_agent(state: State, workflow: Workflow):
#     """
#     Response Generation Agent (STREAMING ENABLED)
#     - CHỈ trả lời dựa trên dữ liệu từ tools
#     - Nếu không có dữ liệu → nói không biết
#     """
#     print("\n[Response Generation] Current State:", state)
#     has_sensor_data = state.get("has_sensor_data", False)
#     has_kb_data = state.get("has_kb_data", False)
#     sensor_data = state.get("sensor_data", {})
#     kb_context = state.get("kb_context", "")
#     qa_feedback = state.get("qa_feedback", {})
#     original_query = state.get("original_query", "")
#     is_small_talk = state.get("is_small_talk", False)
    
#     # Nếu là small talk, trả lời đơn giản
#     if is_small_talk:
#         return {
#             "draft_response": state.get("draft_response", ""),
#             "messages": [AIMessage(content=state.get("draft_response", ""))]
#         }

#     # KIỂM TRA có dữ liệu không
#     if not has_kb_data and not has_sensor_data:
#         # KHÔNG CÓ DỮ LIỆU → Trả lời không biết
#         no_data_response = AIMessage(content=(
#             "Xin lỗi, tôi không tìm thấy thông tin về câu hỏi này trong cơ sở dữ liệu. 🤔\n\n"
#             "Vui lòng hỏi các câu hỏi liên quan đến:\n"
#             "- Kỹ thuật trồng trọt 🌱\n"
#             "- Bón phân và chăm sóc cây 💧\n"
#             "- Kiểm tra dữ liệu cảm biến (nhiệt độ, độ ẩm, NPK, pH) 📊"
#         ))
#         return {
#             "draft_response": no_data_response.content,
#             "messages": [no_data_response]
#         }
    
#     # CÓ DỮ LIỆU → Tạo prompt
#     if has_sensor_data and has_kb_data:
#         system_prompt = (
#             "Bạn là chuyên gia nông nghiệp. CHỈ trả lời dựa trên dữ liệu được cung cấp.\n\n"
#             "LUẬT QUAN TRỌNG:\n"
#             "- CHỈ sử dụng thông tin từ Sensor Data và Knowledge Base Context bên dưới\n"
#             "- KHÔNG bịa thêm thông tin không có trong dữ liệu\n"
#             "- Nếu thiếu thông tin → nói rõ 'cần thêm dữ liệu về...'\n\n"
#             "NHIỆM VỤ:\n"
#             "1. PHÂN TÍCH DỮ LIỆU SENSOR:\n"
#             "   - So sánh với ngưỡng chuẩn từ KB\n"
#             "   - Đánh giá: tốt ✅ / cần cải thiện ⚠️ / nguy hiểm ❌\n"
#             "   - Giải thích nguyên nhân\n\n"
#             "2. KHUYẾN NGHỊ CỤ THỂ:\n"
#             "   - Dựa trên kiến thức từ KB\n"
#             "   - Tên sản phẩm, liều lượng, cách dùng\n"
#             "   - Thời gian thực hiện\n\n"
#             "3. ĐỊNH DẠNG:\n"
#             "   - Dùng emoji\n"
#             "   - Ngắn gọn, dễ hiểu\n"
#             "   - Hành động cụ thể\n"
#         )
#         data_context = f"\n\n--- SENSOR DATA ---\n{json.dumps(sensor_data, ensure_ascii=False, indent=2)}\n\n--- KNOWLEDGE BASE ---\n{kb_context}"
    
#     elif has_kb_data:
#         system_prompt = (
#             "Bạn là chuyên gia nông nghiệp. CHỈ trả lời dựa trên Knowledge Base Context được cung cấp.\n\n"
#             "LUẬT QUAN TRỌNG:\n"
#             "- CHỈ sử dụng thông tin từ Knowledge Base Context bên dưới\n"
#             "- KHÔNG bịa thêm thông tin không có trong KB\n"
#             "- Nếu KB không có đủ thông tin → nói rõ 'không có thông tin về...'\n\n"
#             "NHIỆM VỤ:\n"
#             "1. Trả lời dựa trên KB\n"
#             "2. Đưa ra các bước cụ thể\n"
#             "3. Số liệu chính xác (nếu có)\n"
#             "4. Ngắn gọn, dễ hiểu\n"
#             "5. Dùng emoji để dễ đọc\n"
#         )
#         data_context = f"\n\n--- KNOWLEDGE BASE ---\n{kb_context}"
    
#     else:  # has_sensor_data only
#         system_prompt = (
#             "Bạn là chuyên gia nông nghiệp. CHỈ trả lời dựa trên Sensor Data được cung cấp.\n\n"
#             "LUẬT QUAN TRỌNG:\n"
#             "- CHỈ mô tả dữ liệu sensor\n"
#             "- KHÔNG đưa ra khuyến nghị nếu không có KB context\n"
#             "- Nói rõ 'cần thêm thông tin về ngưỡng chuẩn'\n"
#         )
#         data_context = f"\n\n--- SENSOR DATA ---\n{json.dumps(sensor_data, ensure_ascii=False, indent=2)}"
    
#     # Nếu có feedback từ QA
#     if qa_feedback.get("needs_improvement"):
#         system_prompt += f"\n\nCẢI THIỆN DỰA TRÊN FEEDBACK:\n{qa_feedback.get('suggestions', '')}"
    
#     # QUAN TRỌNG: Chỉ dùng SystemMessage và HumanMessage
#     # KHÔNG dùng ToolMessage vì không có AIMessage với tool_calls đứng trước
#     system = SystemMessage(content=system_prompt)
#     human = HumanMessage(content=f"Câu hỏi: {original_query}{data_context}")
    
#     messages = [system, human]
    
#     # STREAM response
#     try:
#         result = await workflow.llm.ainvoke(messages)
#         print(f"[Response Generation] Generated response.")
#         print(result)
#         return {
#             "draft_response": result.content,
#             "messages": [AIMessage(content=result.content)]
#         }
#     except asyncio.CancelledError:
#         # Khi bị hủy bởi LangGraph/LangSmith, ta không cần xem là lỗi
#         print("Responder task was cancelled (likely by LangGraph).")
#         return None


# async def quality_assurance_agent(state: State, workflow: Workflow):
#     """
#     Quality Assurance Agent
#     - Kiểm tra có trả lời đúng dựa trên dữ liệu không
#     - Có bịa thông tin không
#     """
#     print("\n[Quality Assurance] Current State:", state)
#     system = SystemMessage(content=(
#         "Bạn là QA Agent - kiểm tra chất lượng câu trả lời.\n\n"
#         "TIÊU CHÍ QUAN TRỌNG:\n"
#         "1. ✅ Chính xác: CHỈ dùng thông tin từ tools, KHÔNG bịa\n"
#         "2. ✅ Đầy đủ: Trả lời đủ câu hỏi\n"
#         "3. ✅ Hữu ích: Có khuyến nghị cụ thể (nếu có KB data)\n"
#         "4. ✅ Trung thực: Nếu thiếu data → phải nói rõ\n\n"
#         "CẢI THIỆN NẾU:\n"
#         "- Score < 75\n"
#         "- Có thông tin không có trong tool results\n"
#         "- Thiếu phân tích quan trọng\n\n"
#         "OUTPUT FORMAT:\n"
#         "{\n"
#         "  \"score\": 0-100,\n"
#         "  \"needs_improvement\": true/false,\n"
#         "  \"strengths\": [\"điểm mạnh 1\"],\n"
#         "  \"weaknesses\": [\"điểm yếu 1\"],\n"
#         "  \"suggestions\": \"Cải thiện cụ thể...\"\n"
#         "}\n"
#     ))
    
#     draft_response = state.get("draft_response", "")
#     original_query = state.get("original_query", "")
#     has_sensor_data = state.get("has_sensor_data", False)
#     has_kb_data = state.get("has_kb_data", False)
#     kb_context = state.get("kb_context", "")[:500]  # First 500 chars
    
#     evaluation_prompt = f"""
# CÂU HỎI: {original_query}

# DỮ LIỆU CÓ:
# - Sensor: {'Có' if has_sensor_data else 'Không'}
# - KB: {'Có' if has_kb_data else 'Không'}
# - KB Preview: {kb_context if has_kb_data else 'N/A'}

# CÂU TRẢ LỜI:
# {draft_response}

# Đánh giá và trả về JSON:
# """
    
#     messages = [system, HumanMessage(content=evaluation_prompt)]
#     result = await workflow.llm_router.ainvoke(messages)
    
#     # Parse QA feedback (ROBUST PARSING)
#     try:
#         content = result.content.strip()
        
#         # Remove markdown code blocks
#         if content.startswith("```"):
#             lines = content.split("\n")
#             content = "\n".join(lines[1:-1])  # Remove first and last lines
#             if content.startswith("json"):
#                 content = content[4:].strip()
        
#         # Replace single quotes with double quotes for JSON compatibility
#         content = content.replace("'", '"')
        
#         qa_feedback = json.loads(content.strip())
#     except Exception as e:
#         print(f"[QA] Parse error: {e}")
#         print(f"[QA] Raw content: {result.content[:200]}")
#         qa_feedback = {
#             "score": 85,
#             "needs_improvement": False,
#             "strengths": ["Câu trả lời hợp lý"],
#             "weaknesses": [],
#             "suggestions": ""
#         }
    
#     needs_improvement = qa_feedback.get("needs_improvement", False)
    
#     return {
#         "qa_feedback": qa_feedback,
#         "needs_improvement": needs_improvement,
#         "messages": [AIMessage(content=f"[QA Score: {qa_feedback.get('score')}/100]")]
#     }


# # --------------------------
# # Router Functions
# # --------------------------

# def supervisor_router(state: State) -> str:
#     """Route dựa trên quyết định của supervisor"""
#     print("\n[Router] Supervisor router State:", state)
#     next_agent = state.get("next_agent", "end")
    
#     if next_agent == "end":
#         return "end"
    
#     return next_agent


# # --------------------------
# # Build Graph
# # --------------------------
# def build_graph(workflow: Workflow):
#     graph_builder = StateGraph(State)
    
#     # Add all agent nodes
#     graph_builder.add_node("supervisor", partial(supervisor, workflow=workflow))
#     graph_builder.add_node("query_enhancement", partial(query_enhancement_agent, workflow=workflow))
#     graph_builder.add_node("tool_execution", partial(tool_execution_agent, workflow=workflow))
#     graph_builder.add_node("response_generation", partial(response_generation_agent, workflow=workflow))
#     graph_builder.add_node("quality_assurance", partial(quality_assurance_agent, workflow=workflow))
#     # ✅ FIX 5: Thêm node mới cho small talk
#     graph_builder.add_node("small_talk_response", partial(small_talk_response_agent, workflow=workflow))
    
#     # Set entry point
#     graph_builder.set_entry_point("supervisor")
    
#     # Add conditional routing from supervisor
#     graph_builder.add_conditional_edges(
#         "supervisor",
#         supervisor_router,
#         {
#             "query_enhancement": "query_enhancement",
#             "tool_execution": "tool_execution",
#             "response_generation": "response_generation",
#             "quality_assurance": "quality_assurance",
#             "small_talk_response": "small_talk_response",  # ✅ FIX 6: Thêm route mới
#             "end": END
#         }
#     )
    
#     # Each agent returns to supervisor for next decision
#     graph_builder.add_edge("query_enhancement", "supervisor")
#     graph_builder.add_edge("tool_execution", "supervisor")
#     graph_builder.add_edge("response_generation", "supervisor")
#     graph_builder.add_edge("quality_assurance", "supervisor")
#     graph_builder.add_edge("small_talk_response", "supervisor")  # ✅ FIX 7: Small talk cũng quay về supervisor
    
#     return graph_builder.compile(checkpointer=memory)


# # --------------------------
# # Usage Example
# # --------------------------
# async def run_agent(query: str, retriever, thread_id: str = "default"):
#     """Helper function to run the agent"""
#     workflow = Workflow(retriever, llm, llm_router)
#     graph = build_graph(workflow)
    
#     config = {"configurable": {"thread_id": thread_id}}
    
#     initial_state = {
#         "messages": [HumanMessage(content=query)],
#         "original_query": query,
#         "iteration_count": 0
#     }
    
#     final_state = await graph.ainvoke(initial_state, config)
    
#     # Return final response
#     return final_state["messages"][-1].content