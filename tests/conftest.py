# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

# Mock các object Pydantic trả về từ LLM
# from src.agent.supervisor_multiagent_SOTA import SupervisorDecision, QueryIntent

@pytest.fixture
def mock_workflow():
    """Tạo một Workflow giả để không kết nối DB thật hay OpenAI thật"""
    mock_wf = MagicMock()
    
    # Mock LLM Router
    mock_wf.llm_router = MagicMock()
    
    # Mock LLM Main
    mock_wf.llm = MagicMock()
    mock_wf.llm.ainvoke = AsyncMock(return_value=AIMessage(content="Mocked Answer"))
    
    # Mock Memory Service
    mock_wf.memory_service = MagicMock()
    mock_wf.memory_service.get_profile = AsyncMock(return_value={"name": "Test User", "crop": "Rice"})
    mock_wf.memory_service.update_profile = AsyncMock()
    
    # Mock Tools
    mock_wf.kb_tool = MagicMock()
    mock_wf.kb_tool.ainvoke = AsyncMock(return_value="Mocked KB Result")
    
    return mock_wf

@pytest.fixture
def mock_config(mock_workflow):
    """Tạo config giả chứa workflow"""
    return {"configurable": {"workflow": mock_workflow}}