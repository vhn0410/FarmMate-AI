"""
Improved RAG Agent application (LangGraph + FastAPI) for agriculture.
- Architecture: Planner -> Router -> Tool Node -> Responder
- Fixes: append messages (preserve history), LLM-based router/planner, tool output validation & truncation,
  chunked sensorthings responses, retriever returns docs (not full prompt), and responder assembles final answer.
- Streaming SSE endpoint preserved.

Note: adapt model classes (ChatOpenAI, ChatGoogleGenerativeAI) to your available SDKs.

"""

from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import itertools
import json
import os
import requests
from uuid import uuid4
from dotenv import load_dotenv

# RAG imports (assume these are available in your env)
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

load_dotenv()

# --------------------------
# Configuration
# --------------------------
PERSIST_DIR = "db_agriculture2/chroma_db"
CHUNKS_JSON = "db_agriculture2/chunks_export.json"
SENSORTHINGS_BASE = os.getenv("SENSORTHINGS_BASE", "http://localhost:8026")
OBS_SERVICE = os.getenv("OBS_SERVICE", "http://localhost:8089/ctu/geo/observations/dataStreamIds/latest")
SENSOR_USER_ID = os.getenv("SENSOR_USER_ID", "a2ecd084-3013-4a15-836c-9c0b7be0b320")

# --------------------------
# Memory & Graph State
# --------------------------
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# --------------------------
# Vector store & retrievers (initialize at startup)
# --------------------------
print("🔄 Initializing vector store and retrievers...")
if os.path.exists(PERSIST_DIR):
    from services.load_vector_store import load_vector_store
    vector_db = load_vector_store(PERSIST_DIR)
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 15})
else:
    print("⚠️ Vector DB not found. Vector retriever disabled.")
    vector_db = None
    vector_retriever = None

# Preload chunk documents for BM25
from services.export_chunks_to_json import export_chunk_json_to_document_langchain
documents: List[Document] = export_chunk_json_to_document_langchain(CHUNKS_JSON)

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15

# Ensemble (hybrid) retriever — gracefully degrade if vector_retriever is missing
retrievers = []
weights = []
if vector_retriever:
    retrievers.append(vector_retriever)
    weights.append(0.7)
retrievers.append(bm25_retriever)
weights.append(0.3)

hybrid_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
print("✅ Retrievers setup complete")

# --------------------------
# Utility helpers
# --------------------------

def safe_json_escape(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return (
        text
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\b", "\\b")
        .replace("\f", "\\f")
    )


def serialise_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        content = chunk.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return ''.join(parts)
        if isinstance(content, str):
            return content
        if isinstance(content, dict) and "text" in content:
            return content["text"]
        return str(content)
    raise TypeError("Invalid AIMessageChunk")

# --------------------------
# Validation for sensor values
# --------------------------

def validate_sensor_values(things: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clamp or flag obviously invalid sensor readings (e.g., pH > 14, negative EC when impossible).
    We don't 'fix' values — we annotate them so the LLM sees warnings.
    """
    for t in things:
        for sensor in t.get("sensors", []):
            for ds in sensor.get("datastreams", []):
                obs = ds.get("latest_observation")
                if not obs:
                    continue
                value = obs.get("result")
                # try to handle numeric results
                try:
                    val = float(value)
                    # pH heuristics: if unit contains 'pH' or observedProperty contains 'pH'
                    op = ds.get("observedProperty", {}) or {}
                    op_name = op.get("name", "").lower() if isinstance(op, dict) else ""
                    if "ph" in op_name:
                        if val < 0 or val > 14:
                            obs["_validator"] = f"suspicious_pH_value:{val}"
                    # temperature heuristics
                    if "temp" in op_name or "temperature" in op_name:
                        if val < -50 or val > 70:
                            obs["_validator"] = f"suspicious_temperature:{val}"
                    # EC heuristics (just example)
                    if "ec" in op_name and val < 0:
                        obs["_validator"] = f"negative_ec:{val}"
                except Exception:
                    # not numeric — skip
                    continue
    return things

# --------------------------
# Tools
# --------------------------
@tool
def sensorthings_search(query: str) -> Dict[str, Any]:
    """Return a concise JSON summary of 'things' and their latest observations.
    The tool will not dump massive raw JSON; it returns a list of devices with: id, name, sensor-count, datastream-summary.
    """
    try:
        url_thing = (
            f"{SENSORTHINGS_BASE}/get-things?"
            f"filter=properties/user_id%20eq%20%27{SENSOR_USER_ID}%27"
            "&expand=sensors($expand=datastreams($expand=observedproperties,measurementunits))"
        )
        TOKEN = os.getenv("SENSORTHINGS_TOKEN", "")
        headers = {"token": TOKEN, "Content-Type": "application/json"}
        r = requests.get(url_thing, headers=headers, timeout=10)
        if r.status_code != 200:
            return {"error": True, "message": f"Things API returned {r.status_code}", "data": None}
        things = r.json()
        # flatten
        things = list(itertools.chain.from_iterable(things))
        # get datastream ids
        datastream_ids = []
        for t in things:
            for s in t.get("sensors", []):
                for ds in s.get("datastreams", []):
                    datastream_ids.append(str(ds.get("id")))
        # observations
        obs_payload = json.dumps(datastream_ids)
        r2 = requests.post(OBS_SERVICE, headers={"accept": "application/json", "Content-Type": "application/json"}, data=obs_payload, timeout=10)
        if r2.status_code != 200:
            return {"error": True, "message": f"Observations API returned {r2.status_code}", "data": None}
        observations = r2.json()
        obs_map = {item["dataStreamId"]: item for item in observations}
        for t in things:
            for s in t.get("sensors", []):
                for ds in s.get("datastreams", []):
                    ds_id = str(ds.get("id"))
                    ds["latest_observation"] = obs_map.get(ds_id)
        # validate and annotate suspicious values
        things = validate_sensor_values(things)
        # build condensed summary to keep LLM tractable
        summary = []
        for t in things:
            t_summary = {"id": t.get("id"), "name": t.get("name"), "sensors": []}
            for s in t.get("sensors", []):
                s_summary = {"id": s.get("id"), "name": s.get("name"), "datastreams": []}
                for ds in s.get("datastreams", []):
                    ds_summary = {
                        "id": ds.get("id"),
                        "name": ds.get("name"),
                        "observedProperty": ds.get("observedProperty"),
                        "unit": ds.get("measurementUnit"),
                        "latest_observation": ds.get("latest_observation") and {
                            "result": ds.get("latest_observation").get("result"),
                            "phenomenonTime": ds.get("latest_observation").get("phenomenonTime"),
                            "_validator": ds.get("latest_observation", {}).get("_validator")
                        }
                    }
                    s_summary["datastreams"].append(ds_summary)
                t_summary["sensors"].append(s_summary)
            summary.append(t_summary)
        return {"error": False, "message": "success", "data": summary}
    except Exception as e:
        return {"error": True, "message": str(e), "data": None}


@tool
def search_knowledge_base(query: str) -> str:
    """Retrieve top documents from hybrid retriever, rerank, and return structured JSON with short excerpts.
    The tool returns a JSON string with a list of docs: {title, excerpt, source, doc_id}.
    """
    try:
        retrieved_docs = hybrid_retriever.invoke(query)
        reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
        reranked = reranker.compress_documents(retrieved_docs, query)
        if not reranked:
            return json.dumps({"type": "text_only", "text": "Không tìm thấy thông tin phù hợp."}, ensure_ascii=False)
        docs_out = []
        for d in reranked[:8]:
            text_excerpt = (d.page_content[:600] + "...") if len(d.page_content) > 600 else d.page_content
            docs_out.append({"title": d.metadata.get("title", ""), "excerpt": text_excerpt, "source": d.metadata.get("source", ""), "id": d.metadata.get("id", "")})
        return json.dumps({"type": "docs", "query": query, "results": docs_out}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"type": "error", "message": str(e)})

# --------------------------
# LLM setup
# --------------------------
# Choose whichever LLM is available in your env. We'll make two llms: one for planning/router (no tools) and one bound-to-tools.
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # optional smaller router

# Bind tools for the model that will do generation with tool-calls
llm_with_tools = llm.bind_tools(tools=[search_knowledge_base, sensorthings_search])

# --------------------------
# Nodes: planner, tool_node, responder
# --------------------------
async def planner(state: State):
    """Planner decides whether to call tools and with what queries. It should not produce final user-facing answer.
    Output: appended AI planning message (may include tool_calls).
    """
    # Build a short system prompt to the model
    system = SystemMessage(content=(
        "You are a Planner that decides whether to call tools.\n"
        "If user asks about devices, sensors, datastreams, or measurements --> call sensorthings_search.\n"
        "If user asks about agronomy/treatment/recommendations --> call search_knowledge_base.\n"
        "Return only a plan or tool call, do NOT produce the final answer."
    ))

    # Send planner message to llm_with_tools so it can emit tool_calls
    messages = [system] + state["messages"]
    result = await llm_with_tools.ainvoke(messages)
    # append (preserve history)
    return {"messages": state["messages"] + [result]}


async def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return "responder"


async def tool_node(state: State):
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
                        text = f"[KB Results for: {parsed.get('query')} ]\\n"
                        for r in parsed.get("results", []):
                            text += f"- {r.get('title')} | {r.get('source')} | {r.get('excerpt')}\\n"
                    else:
                        text = parsed.get("text") if parsed.get("text") else str(parsed)
                except Exception:
                    text = out
                # tool_messages.append(ToolMessage(content=text, tool_call_id=tool_id, name=name))
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
                    tool_messages.append(ToolMessage(content=pretty, tool_call_id=tool_id, name=name))
                else:
                    tool_messages.append(ToolMessage(content=str(out), tool_call_id=tool_id, name=name))
            else:
                tool_messages.append(ToolMessage(content=f"Unknown tool: {name}", tool_call_id=tool_id, name=name))
        except Exception as e:
            tool_messages.append(ToolMessage(content=f"Tool {name} error: {str(e)}", tool_call_id=tool_id, name=name))
    # Return appended messages (so history preserved)
    return {"messages": state["messages"] + tool_messages}


async def responder(state: State):
    """Responder: use the latest context (including tool messages) to produce final answer to user.
    Append the response and preserve history.
    """
    # small system prompt to instruct the model to use the documents/tools
    system = SystemMessage(content=(
        "You are an expert agricultural assistant. Use the retrieved tool outputs and knowledge-base excerpts to produce a concise, actionable answer for a farmer.\n"
        "If the tool outputs include suspicious sensor values (marked with _validator), call that out and recommend verification steps.\n"
        "If data is insufficient, be explicit and ask for specific extra info.\n"
        "Provide numbers where possible and short practical steps."
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
graph_builder.set_entry_point("planner")
graph_builder.add_conditional_edges("planner", tools_router)
graph_builder.add_edge("tool_node", "responder")
graph = graph_builder.compile(checkpointer=memory)

# --------------------------
# FastAPI & SSE
# --------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    checkpoint_id: Optional[str] = None


async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    try:
        is_new = checkpoint_id is None
        if is_new:
            checkpoint = str(uuid4())
            config = {"configurable": {"thread_id": checkpoint}}
            yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{checkpoint}\"}}\n\n"
        else:
            config = {"configurable": {"thread_id": checkpoint_id}}

        events = graph.astream_events({"messages": [HumanMessage(content=message)]}, version="v2", config=config)

        async for event in events:
            et = event["event"]
            if et == "on_chat_model_stream":
                try:
                    chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
                    if not chunk_content:
                        continue
                    safe = safe_json_escape(chunk_content)
                    yield f"data: {{\"type\": \"content\", \"content\": \"{safe}\"}}\n\n"
                except Exception as e:
                    continue
            elif et == "on_chat_model_end":
                try:
                    output = event["data"]["output"]
                    tool_calls = getattr(output, "tool_calls", [])
                    rag_calls = [c for c in tool_calls if c["name"] == "search_knowledge_base"]
                    if rag_calls:
                        q = rag_calls[0]["args"].get("query", "")
                        yield f"data: {{\"type\": \"rag_search_start\", \"query\": \"{safe_json_escape(str(q))}\"}}\n\n"
                except Exception:
                    pass
            elif et == "on_tool_end":
                try:
                    name = event.get("name")
                    output = event["data"]["output"]
                    preview = str(output)
                    if len(preview) > 500:
                        preview = preview[:500] + "..."
                    yield f"data: {{\"type\": \"tool_complete\", \"tool\": \"{name}\", \"preview\": \"{safe_json_escape(preview)}\"}}\n\n"
                except Exception:
                    pass
        yield f"data: {{\"type\": \"end\"}}\n\n"
    except Exception as e:
        yield f"data: {{\"type\": \"error\", \"message\": \"{safe_json_escape(str(e))}\"}}\n\n"


# @app.post("/chat_stream")
# async def chat_stream_post(req: ChatRequest):
#     return StreamingResponse(generate_chat_responses(req.message, req.checkpoint_id), media_type="text/event-stream")

@app.get("/chat_stream")
async def chat_stream_get(message: str, checkpoint_id: Optional[str] = Query(None)):
    """GET endpoint - legacy support. Use POST for better reliability."""
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/health")
async def health():
    return {"status": "ok", "rag_enabled": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
