"""
Test script để kiểm tra memory trong multi-turn conversation
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)

def send_message(message: str, checkpoint_id: str = None):
    """
    Gửi message và nhận streaming response
    """
    if checkpoint_id:
        url = f"{BASE_URL}/chat_stream?message={requests.utils.quote(message)}&checkpoint_id={checkpoint_id}"
    else:
        url = f"{BASE_URL}/chat_stream?message={requests.utils.quote(message)}"
    
    print(f"\n🗣️  User: {message}")
    print(f"🤖 Bot: ", end="", flush=True)
    
    response = requests.get(url, stream=True)
    new_checkpoint = checkpoint_id
    full_response = ""
    
    for line in response.iter_lines():
        if line:
            data = line.decode().replace("data: ", "")
            try:
                event = json.loads(data)
                
                if event["type"] == "checkpoint":
                    new_checkpoint = event["checkpoint_id"]
                    print(f"\n   [Checkpoint: {new_checkpoint[:8]}...]")
                    print(f"🤖 Bot: ", end="", flush=True)
                
                elif event["type"] == "content":
                    content = event["content"]
                    print(content, end="", flush=True)
                    full_response += content
                
                elif event["type"] == "end":
                    print()  # New line after response
                
                elif event["type"] == "error":
                    print(f"\n❌ Error: {event['message']}")
            except json.JSONDecodeError:
                pass
    
    return new_checkpoint, full_response


def test_memory_conversation():
    """
    Test case: Memory trong multi-turn conversation
    """
    print_separator("🧪 MEMORY TEST - Multi-Turn Conversation")
    
    checkpoint = None
    
    # Turn 1: Giới thiệu tên
    print_separator("TURN 1: Giới thiệu")
    checkpoint, _ = send_message("xin chào, tôi tên là Hoàng Vũ", checkpoint)
    time.sleep(1)
    
    # Turn 2: Hỏi lại tên (kiểm tra memory)
    print_separator("TURN 2: Kiểm tra memory - Tôi là ai?")
    checkpoint, response2 = send_message("tôi là ai?", checkpoint)
    time.sleep(1)
    
    # Turn 3: Hỏi câu trước (kiểm tra conversation history)
    print_separator("TURN 3: Kiểm tra conversation history")
    checkpoint, response3 = send_message("tôi vừa hỏi gì?", checkpoint)
    time.sleep(1)
    
    # Turn 4: Giới thiệu thêm thông tin
    print_separator("TURN 4: Thêm thông tin")
    checkpoint, _ = send_message("tôi đang trồng lúa ở Cần Thơ", checkpoint)
    time.sleep(1)
    
    # Turn 5: Hỏi lại thông tin vừa chia sẻ
    print_separator("TURN 5: Kiểm tra long-term memory")
    checkpoint, response5 = send_message("tôi đang trồng gì?", checkpoint)
    time.sleep(1)
    
    # Turn 6: Context-aware question
    print_separator("TURN 6: Context-aware query")
    checkpoint, response6 = send_message("nó phù hợp với khí hậu ở đó không?", checkpoint)
    
    # Validation
    print_separator("📊 MEMORY VALIDATION")
    
    checks = [
        ("Turn 2: Nhớ tên 'Hoàng Vũ'", "hoàng vũ" in response2.lower() or "hoang vu" in response2.lower()),
        ("Turn 3: Nhớ câu hỏi trước", "tôi là ai" in response3.lower() or "là ai" in response3.lower()),
        ("Turn 5: Nhớ thông tin cây trồng", "lúa" in response5.lower()),
        ("Turn 6: Hiểu context 'nó'", "lúa" in response6.lower() or "cần thơ" in response6.lower()),
    ]
    
    passed = 0
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{len(checks)} tests passed")
    
    if passed == len(checks):
        print("🎉 MEMORY WORKS PERFECTLY!")
    elif passed >= len(checks) // 2:
        print("⚠️  MEMORY PARTIALLY WORKING")
    else:
        print("❌ MEMORY NOT WORKING")
    
    return checkpoint


def test_context_switch():
    """
    Test case: Chuyển đổi giữa small talk và agricultural queries
    """
    print_separator("🧪 CONTEXT SWITCH TEST")
    
    checkpoint = None
    
    # Small talk
    print_separator("PHASE 1: Small Talk")
    checkpoint, _ = send_message("xin chào", checkpoint)
    time.sleep(1)
    
    # Agricultural query
    print_separator("PHASE 2: Agricultural Query")
    checkpoint, _ = send_message("cách trồng lúa hiệu quả", checkpoint)
    time.sleep(1)
    
    # Back to small talk (should still remember previous context)
    print_separator("PHASE 3: Back to Small Talk (with memory)")
    checkpoint, response3 = send_message("cảm ơn bạn nhé", checkpoint)
    time.sleep(1)
    
    # Context-aware query
    print_separator("PHASE 4: Context-aware Query")
    checkpoint, response4 = send_message("còn gì khác không?", checkpoint)
    
    print_separator("✅ CONTEXT SWITCH TEST COMPLETED")
    return checkpoint


def test_new_conversation():
    """
    Test case: Conversation mới không bị nhiễm data cũ
    """
    print_separator("🧪 NEW CONVERSATION TEST")
    
    # Conversation 1
    print_separator("CONVERSATION 1")
    checkpoint1 = None
    checkpoint1, _ = send_message("tôi tên là Alice", checkpoint1)
    time.sleep(1)
    
    # Conversation 2 (new checkpoint - should NOT remember Alice)
    print_separator("CONVERSATION 2 (New Thread)")
    checkpoint2 = None
    checkpoint2, response2 = send_message("tôi là ai?", checkpoint2)
    
    # Validation
    print_separator("📊 ISOLATION VALIDATION")
    if "alice" not in response2.lower():
        print("✅ PASS - New conversation does NOT remember old data")
    else:
        print("❌ FAIL - New conversation leaked data from old thread")
    
    return checkpoint1, checkpoint2


if __name__ == "__main__":
    try:
        # Check if server is running
        health = requests.get(f"{BASE_URL}/health", timeout=2)
        if health.status_code != 200:
            print("❌ Server is not healthy!")
            exit(1)
        
        print("=" * 70)
        print("  🚀 AGRICULTURAL AI AGENT - MEMORY TEST SUITE")
        print("=" * 70)
        print(f"  Server: {BASE_URL}")
        print(f"  Status: {health.json()['status']}")
        print("=" * 70)
        
        # Run tests
        test_memory_conversation()
        print("\n" + "=" * 70 + "\n")
        
        test_context_switch()
        print("\n" + "=" * 70 + "\n")
        
        test_new_conversation()
        
        print_separator("🎉 ALL TESTS COMPLETED!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start the server first:")
        print("   python app.py")
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")