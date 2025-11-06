"""
Test script for the RAG-enabled LangGraph system
Run this to test the RAG tool directly before integrating with FastAPI
"""

from services.load_vector_store import load_vector_store
from services.ingesion_pipeline import run_complete_ingestion_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

# ===== Define the RAG Tool =====
@tool
async def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for information about the Transformer architecture.
    
    Args:
        query: The search query
    
    Returns:
        Relevant text chunks from the knowledge base
    """
    persist_directory = "dbv2/chroma_db"
    
    if os.path.exists(persist_directory):
        print("📚 Loading existing ChromaDB...")
        db = load_vector_store(persist_directory)
    else:
        print("🚀 Creating new ChromaDB and embedding documents...")
        db = run_complete_ingestion_pipeline(
            pdf_path="./docs/attention-is-all-you-need.pdf"
        )
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke(query)
    
    if not chunks:
        return "No relevant information found in the knowledge base."
    
    formatted_results = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.page_content
        metadata = chunk.metadata
        formatted_results.append(
            f"[Chunk {i}]\n{content}\n"
            f"Source: Page {metadata.get('page', 'Unknown')}\n"
        )
    
    return "\n---\n".join(formatted_results)


async def test_direct_rag():
    """Test RAG tool directly"""
    print("\n" + "="*80)
    print("TEST 1: Direct RAG Tool Test")
    print("="*80)
    
    query = "What are the two main components of the Transformer architecture?"
    print(f"\nQuery: {query}\n")
    
    result = await search_knowledge_base.ainvoke({"query": query})
    print("Result:")
    print(result)


async def test_with_llm():
    """Test RAG tool with LLM integration"""
    print("\n" + "="*80)
    print("TEST 2: LLM + RAG Tool Integration")
    print("="*80)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    llm_with_tools = llm.bind_tools(tools=[search_knowledge_base])
    
    # Test queries
    queries = [
        "What are the main components of the Transformer?",
        "Explain multi-head attention mechanism",
        "What is the weather today?"  # Should not trigger RAG
    ]
    
    for query in queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print('─'*80)
        
        response = await llm_with_tools.ainvoke(query)
        
        # Check if tool was called
        if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
            print(f"✅ LLM decided to use RAG tool")
            print(f"Tool call: {response.tool_calls[0]['name']}")
            print(f"Arguments: {response.tool_calls[0]['args']}")
            
            # Execute the tool
            tool_result = await search_knowledge_base.ainvoke(
                response.tool_calls[0]['args']
            )
            print(f"\n📖 RAG Results (truncated):")
            print(tool_result[:300] + "...")
        else:
            print(f"❌ LLM did not use RAG tool")
            print(f"Response: {response.content}")


async def test_conversation_flow():
    """Test a multi-turn conversation with RAG"""
    print("\n" + "="*80)
    print("TEST 3: Multi-turn Conversation with RAG")
    print("="*80)
    
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    llm_with_tools = llm.bind_tools(tools=[search_knowledge_base])
    
    # Simulate a conversation
    messages = [
        HumanMessage(content="What is the Transformer architecture?")
    ]
    
    print("\n👤 User: What is the Transformer architecture?")
    
    # First LLM response
    ai_response = await llm_with_tools.ainvoke(messages)
    messages.append(ai_response)
    
    if hasattr(ai_response, "tool_calls") and len(ai_response.tool_calls) > 0:
        print("🤖 Assistant: [Using RAG to search knowledge base...]")
        
        # Execute tool
        tool_result = await search_knowledge_base.ainvoke(
            ai_response.tool_calls[0]['args']
        )
        
        # Add tool result to conversation
        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=ai_response.tool_calls[0]['id'],
            name="search_knowledge_base"
        )
        messages.append(tool_message)
        
        # Get final response
        final_response = await llm_with_tools.ainvoke(messages)
        print(f"🤖 Assistant: {final_response.content}")
    else:
        print(f"🤖 Assistant: {ai_response.content}")


if __name__ == "__main__":
    import asyncio
    
    # Run all tests
    async def run_all_tests():
        await test_direct_rag()
        await test_with_llm()
        await test_conversation_flow()
    
    asyncio.run(run_all_tests())