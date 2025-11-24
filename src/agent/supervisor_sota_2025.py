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
    """🔥 NEW: Explicit execution plan"""
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
    
    system_prompt = f"""You are an intent classifier for agricultural AI assistant.

USER PROFILE:
- Name: {profile.get('name', 'Unknown')}
- Crops: {', '.join(profile.get('crop_types', ['Not specified']))}

CLASSIFY INTO:
1. **small_talk**: Greetings, thanks.
2. **knowledge**: General farming advice (e.g., "How to plant rice?").
3. **sensor**: User ONLY wants raw numbers (e.g., "What is the temp?", "Show me logs").
4. **consultation**: User wants ANALYSIS, DIAGNOSIS, or CHECK STATUS (e.g., "Is my soil good?", "Check nutrition", "Plant is yellow").

CRITICAL RULES:
- "kiểm tra đất/dinh dưỡng" (Check soil/nutrition) → "consultation" (Because we need to Analyze if the data is good/bad using KB).
- "nhiệt độ bao nhiêu" (What is temp) → "sensor" (Just give the number).
- "cây bị bệnh gì" (What disease) → "consultation" (Sensor + KB).

Return JSON."""
    
    router = wf.llm_fast.with_structured_output(IntentResult)
    result = await router.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
    print(f"🔍 Intent: {result.intent} ({result.confidence:.0%}) - {result.reasoning}")
    
    # Route based on intent
    if result.intent == "small_talk":
        next_worker = "small_talk_worker"
    elif result.intent in ["sensor", "consultation"]:
        next_worker = "planner_node"  # 🔥 NEW: Plan execution
    else:  # knowledge
        next_worker = "retrieval_worker"  # Direct KB retrieval
    
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
    
    system_prompt = f"""You are an execution planner.

INTENT: {intent}
QUERY: {query}

Create a plan to answer the user.

STRATEGY:
1. If the user asks to "Check", "Analyze", or "Diagnose" (Consultation):
   - We MUST get current status first -> ["sensor_tool"]
   - THEN we MUST consult the Knowledge Base to interpret those numbers -> ["kb_tool"]
   - Plan: ["sensor_tool", "kb_tool"]

2. If the user just asks for "Current reading" or "Log" (Sensor):
   - Plan: ["sensor_tool"]

3. If the user asks "How to" or "Theory" (Knowledge):
   - Plan: ["kb_tool"]

OUTPUT INSTRUCTION:
- For "Check nutrition/soil":
  - steps: ["sensor_tool", "kb_tool"]
  - sensor_query: "Get NPK, pH, moisture levels"
  - kb_query: "Standard nutrient levels for rice/fruit trees, diagnosis based on sensor data" (The system will inject sensor data automatically).

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
        result = await sensorthings_search.ainvoke(sensor_query)
        
        if result and isinstance(result, dict):
            sensor_summary = json.dumps(result, ensure_ascii=False, indent=2)
            tool_msg = ToolMessage(
                content=f"[SENSOR DATA]\n{sensor_summary}",
                tool_call_id="sensor_tool"
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
            print("⚠️ Sensors: No data")
            return {
                "messages": [ToolMessage(
                    content="[SENSOR DATA] No sensor data available",
                    tool_call_id="sensor_tool"
                )],
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
            tool_msg = ToolMessage(
                content=f"[KNOWLEDGE BASE]\n{result}",
                tool_call_id="kb_tool"
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
            return {
                "messages": [ToolMessage(
                    content="[KNOWLEDGE BASE] No relevant information found",
                    tool_call_id="kb_tool"
                )],
                "current_step": current_step + 1,
                "next_worker": "executor_node"
            }


# ==========================================
# 7. SMALL TALK WORKER
# ==========================================

async def small_talk_worker(state: AgentState, config: RunnableConfig):
    wf = config["configurable"]["workflow"]
    query = state["messages"][-1].content
    profile = state["user_profile"]
    
    name = profile.get("name", "bạn")
    
    system_prompt = f"""You are a friendly agricultural assistant chatting with {name}.

Keep responses:
- Warm and encouraging
- Brief (2-3 sentences)
- Subtly mention what you can help with (sensors, farming advice)

Use emojis naturally 🌾🚜"""
    
    response = await wf.llm_fast.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
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
        tool_msg = ToolMessage(
            content=f"[KNOWLEDGE BASE]\n{result}",
            tool_call_id="kb_tool"
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
            "messages": [ToolMessage(
                content="No relevant information found",
                tool_call_id="kb_tool"
            )],
            "next_worker": "synthesis_worker"
        }


# ==========================================
# 9. SYNTHESIS WORKER
# ==========================================

async def synthesis_worker(state: AgentState, config: RunnableConfig):
    wf = config["configurable"]["workflow"]
    query = state["messages"][-1].content
    profile = state["user_profile"]
    
    # Extract tool results
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    
    if not tool_messages:
        no_data_msg = AIMessage(content=(
            "Xin lỗi, tôi không tìm thấy thông tin về câu hỏi này. "
            "Hãy hỏi về kỹ thuật canh tác, dữ liệu cảm biến, hoặc vấn đề cây trồng nhé! 🌱"
        ))
        return {
            "messages": [no_data_msg],
            "next_worker": "END"
        }
    
    # Build context
    context_parts = []
    has_sensor = state.get("sensor_data") is not None
    has_kb = state.get("kb_context") is not None
    
    for tm in tool_messages:
        context_parts.append(tm.content)
    context = "\n\n".join(context_parts)
    
    # Adjusted prompt based on data type
    if has_sensor and has_kb:
        instruction = """You have BOTH sensor readings and farming knowledge.

STRUCTURE YOUR RESPONSE:
1. **Current Status** 🌱 (from sensors)
   - List key sensor readings
   
2. **Analysis** 🔍
   - Interpret sensor values using knowledge base
   - Identify issues or patterns
   
3. **Recommendations** 🌾
   - Specific actions based on sensor + knowledge
   - Use farming best practices from KB
   
4. **Next Steps** 📝
   - Immediate actions
   - Monitoring plan"""
    
    elif has_sensor:
        instruction = """You have sensor data only.

STRUCTURE:
1. Current readings
2. What they mean
3. General recommendations"""
    
    else:
        instruction = """You have farming knowledge only.

Provide clear, actionable advice with specific steps."""
    
    system_prompt = f"""You are an expert agricultural consultant helping {profile.get('name', 'farmer')}.

GUIDELINES:
1. **ONLY use information from the provided data**
2. **DO NOT make up information**
3. **Cite sources** when using sensor readings or specific facts
4. **Be actionable**: Give specific steps, amounts, timings
5. **Use emojis** to make it friendly 🌱🔍🌾📝

{instruction}

CONTEXT FROM TOOLS:
{context}
"""
    
    # Stream response
    response_chunks = []
    async for chunk in wf.llm_smart.astream([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {query}")
    ]):
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
    reflection_prompt = f"""Evaluate this agricultural AI response:

QUERY: {query}
RESPONSE: {response}
AVAILABLE CONTEXT: {context[:300]}...

CRITERIA:
1. Accuracy: Only uses provided data? (0-30)
2. Completeness: Answers the question? (0-30)
3. Actionability: Gives specific advice? (0-20)
4. Clarity: Easy to understand? (0-20)

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