# tests/test_supervisor_logic.py
import pytest
from unittest.mock import AsyncMock
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.supervisor_multiagent_SOTA import supervisor_node, SupervisorDecision

@pytest.mark.asyncio
async def test_supervisor_initial_state(mock_config):
    """Test 1: Mới bắt đầu hoặc chưa có profile -> Phải đi Memory Worker"""
    state = {
        "messages": [HumanMessage(content="Xin chào")],
        "user_profile": {}, # Profile rỗng
        "last_node": "start"
    }
    
    result = await supervisor_node(state, mock_config)
    assert result["next_node"] == "memory_worker"

@pytest.mark.asyncio
async def test_supervisor_routing_chat(mock_config):
    """Test 2: Nếu AI Router bảo là chat -> Đi chat_worker"""
    state = {
        "messages": [HumanMessage(content="Chào bạn")],
        "user_profile": {"name": "Test"},
        "last_node": "memory_worker" # Giả sử đã load memory xong
    }
    
    # --- MOCKING LLM ROUTER ---
    # Giả lập LLM trả về quyết định là 'chat_worker'
    mock_router_chain = AsyncMock()
    mock_router_chain.ainvoke.return_value = SupervisorDecision(
        next_node="chat_worker", 
        reasoning="User is greeting"
    )
    
    # Gán mock vào workflow
    wf = mock_config["configurable"]["workflow"]
    wf.llm_router.with_structured_output.return_value = mock_router_chain
    
    # Chạy hàm
    result = await supervisor_node(state, mock_config)
    
    # Kiểm tra kết quả
    assert result["next_node"] == "chat_worker"

@pytest.mark.asyncio
async def test_supervisor_routing_research(mock_config):
    """Test 3: Nếu AI Router bảo là research -> Đi research_worker"""
    state = {
        "messages": [HumanMessage(content="Giá lúa hôm nay?")],
        "user_profile": {"name": "Test"},
        "last_node": "memory_worker"
    }
    
    mock_router_chain = AsyncMock()
    mock_router_chain.ainvoke.return_value = SupervisorDecision(
        next_node="research_worker", 
        reasoning="Checking price requires tools"
    )
    
    wf = mock_config["configurable"]["workflow"]
    wf.llm_router.with_structured_output.return_value = mock_router_chain
    
    result = await supervisor_node(state, mock_config)
    assert result["next_node"] == "research_worker"

@pytest.mark.asyncio
async def test_supervisor_finish_loop(mock_config):
    """Test 4: Nếu worker vừa làm xong -> Phải trả về 'end'"""
    # Trường hợp 1: Chat worker vừa xong
    state_chat = {
        "messages": [], 
        "user_profile": {"name": "Test"},
        "last_node": "chat_worker" # <-- Vừa xong việc
    }
    result = await supervisor_node(state_chat, mock_config)
    assert result["next_node"] == "end"

    # Trường hợp 2: Research worker vừa xong
    state_research = {
        "messages": [], 
        "user_profile": {"name": "Test"},
        "last_node": "research_worker" # <-- Vừa xong việc
    }
    result_2 = await supervisor_node(state_research, mock_config)
    assert result_2["next_node"] == "end"