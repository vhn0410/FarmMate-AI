# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app import app # Import app từ file app.py của bạn

client = TestClient(app)

# Dữ liệu test
valid_payload = {
    "query": "Giá lúa bao nhiêu?",
    "user_id": "test_user_123",
    "thread_id": "thread_abc"
}

def test_health_check_implicit():
    """Test xem app có khởi động được không (vào docs chẳng hạn)"""
    response = client.get("/docs")
    assert response.status_code == 200

@patch("app.app_graph") # Mock cái biến global app_graph trong app.py
def test_chat_endpoint_success(mock_graph):
    """Test trường hợp chat thành công"""
    
    # 1. Setup Mock Return cho Graph
    # Giả lập graph trả về state cuối cùng
    mock_final_state = {
        "final_response": "Giá lúa là 8000đ/kg",
        "messages": [],
        "user_id": "test_user_123"
    }
    
    # Vì app gọi await app_graph.ainvoke -> cần mock là AsyncMock
    mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)
    
    # 2. Gọi API
    response = client.post("/chat", json=valid_payload)
    
    # 3. Assertions
    assert response.status_code == 200
    data = response.json()
    
    assert data["response"] == "Giá lúa là 8000đ/kg"
    assert data["user_id"] == "test_user_123"
    assert data["thread_id"] == "thread_abc"

@patch("app.app_graph")
def test_chat_endpoint_fallback_response(mock_graph):
    """Test trường hợp fallback: worker quên set final_response thì lấy message cuối"""
    
    from langchain_core.messages import AIMessage
    
    mock_final_state = {
        "final_response": None, # Giả sử quên set cái này
        "messages": [AIMessage(content="Câu trả lời từ message cuối")],
        "user_id": "test_user_123"
    }
    
    mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)
    
    response = client.post("/chat", json=valid_payload)
    
    assert response.status_code == 200
    assert response.json()["response"] == "Câu trả lời từ message cuối"

def test_chat_endpoint_error_no_graph():
    """Test lỗi khi Graph chưa khởi tạo (app_graph = None)"""
    # Vì TestClient không kích hoạt lifespan event tự động hoàn toàn như uvicorn,
    # mặc định app_graph trong app.py sẽ là None nếu mình không set.
    
    # Đảm bảo app_graph là None
    with patch("app.app_graph", None):
        response = client.post("/chat", json=valid_payload)
        assert response.status_code == 500
        assert "Agent chưa khởi tạo" in response.json()["detail"]