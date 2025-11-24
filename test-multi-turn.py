"""
🧪 MEMORY TEST SUITE - MongoDB Store Validation
Test cross-thread memory persistence
"""

import asyncio
import requests
import json
from typing import Optional
import time

# Server configuration
BASE_URL = "http://localhost:8000"

def print_header(text: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_subheader(text: str):
    """Print formatted subsection header"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")

def stream_chat(message: str, user_id: str, checkpoint_id: Optional[str] = None):
    """
    Stream chat response from API
    Returns (response_text, checkpoint_id)
    """
    url = f"{BASE_URL}/chat_stream"
    params = {
        "message": message,
        "user_id": user_id
    }
    if checkpoint_id:
        params["checkpoint_id"] = checkpoint_id
    
    response_text = ""
    new_checkpoint = checkpoint_id
    
    try:
        response = requests.get(url, params=params, stream=True, timeout=60)
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    data_str = decoded[6:]  # Remove 'data: ' prefix
                    try:
                        data = json.loads(data_str)
                        
                        if data.get("type") == "checkpoint":
                            new_checkpoint = data.get("checkpoint_id")
                            print(f"   [Checkpoint: {new_checkpoint[:8]}...]")
                        
                        elif data.get("type") == "content":
                            content = data.get("content", "")
                            response_text += content
                            print(content, end='', flush=True)
                        
                        elif data.get("type") == "end":
                            print()  # Newline after response
                            break
                        
                        elif data.get("type") == "error":
                            print(f"\n❌ Error: {data.get('message')}")
                            break
                    
                    except json.JSONDecodeError:
                        pass
        
        return response_text, new_checkpoint
    
    except Exception as e:
        print(f"❌ Request error: {e}")
        return "", checkpoint_id

def get_user_memory(user_id: str):
    """Get user's long-term memory from API"""
    try:
        response = requests.get(f"{BASE_URL}/user_memory/{user_id}", timeout=10)
        data = response.json()
        
        if data.get("success"):
            return data.get("memory", {})
        else:
            return {}
    except Exception as e:
        print(f"❌ Error getting memory: {e}")
        return {}

def delete_user_memory(user_id: str):
    """Delete user's memory"""
    try:
        response = requests.delete(f"{BASE_URL}/user_memory/{user_id}", timeout=10)
        return response.json()
    except Exception as e:
        print(f"❌ Error deleting memory: {e}")
        return {}

def check_health():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        data = response.json()
        return data.get("status") == "ok"
    except:
        return False

# --------------------------
# Test Functions
# --------------------------

def test_memory_persistence():
    """
    Test 1: Memory persists across different threads
    """
    print_header("🧪 TEST 1: MEMORY PERSISTENCE ACROSS THREADS")
    
    user_id = "test_user_hoang_vu"
    
    # Clean slate
    print("🧹 Cleaning old memory...")
    delete_user_memory(user_id)
    time.sleep(1)
    
    # Turn 1: User introduces themselves
    print_subheader("TURN 1: User Introduction")
    print(f"🗣️  User: xin chào, tôi tên là Hoàng Vũ")
    print("🤖 Bot: ")
    response1, checkpoint1 = stream_chat(
        message="xin chào, tôi tên là Hoàng Vũ",
        user_id=user_id
    )
    
    # Turn 2: Add crop info (same thread)
    print_subheader("TURN 2: Add Crop Info (Same Thread)")
    print(f"🗣️  User: tôi đang trồng lúa ở Cần Thơ")
    print("🤖 Bot: ")
    response2, checkpoint1 = stream_chat(
        message="tôi đang trồng lúa ở Cần Thơ",
        user_id=user_id,
        checkpoint_id=checkpoint1
    )
    
    # Check memory
    print_subheader("MEMORY CHECK")
    memory = get_user_memory(user_id)
    print(f"📋 User Memory:")
    print(f"   Name: {memory.get('name', 'Not set')}")
    print(f"   Crop: {memory.get('crop_type', 'Not set')}")
    print(f"   Location: {memory.get('location', 'Not set')}")
    
    # Turn 3: NEW THREAD - Check if memory persists
    print_subheader("TURN 3: NEW THREAD - Memory Test")
    print(f"🗣️  User: tôi là ai?")
    print("🤖 Bot: ")
    response3, checkpoint2 = stream_chat(
        message="tôi là ai?",
        user_id=user_id
        # Note: No checkpoint_id = new thread!
    )
    
    # Validation
    print_subheader("📊 VALIDATION")
    
    # Check if bot remembered the name
    success_name = "Hoàng Vũ" in response3 or "hoàng vũ" in response3.lower()
    
    if success_name:
        print("✅ PASS - Bot remembers user name across threads!")
    else:
        print("❌ FAIL - Bot forgot user name")
        print(f"   Expected: Contains 'Hoàng Vũ'")
        print(f"   Got: {response3[:100]}...")
    
    # Turn 4: Another NEW THREAD - Check crop memory
    print_subheader("TURN 4: ANOTHER NEW THREAD - Crop Test")
    print(f"🗣️  User: tôi đang trồng gì?")
    print("🤖 Bot: ")
    response4, checkpoint3 = stream_chat(
        message="tôi đang trồng gì?",
        user_id=user_id
        # Another new thread!
    )
    
    success_crop = "lúa" in response4.lower()
    success_location = "cần thơ" in response4.lower()
    
    if success_crop and success_location:
        print("\n✅ PASS - Bot remembers crop and location!")
    elif success_crop:
        print("\n⚠️  PARTIAL - Bot remembers crop but not location")
    else:
        print("\n❌ FAIL - Bot forgot crop information")
    
    return all([success_name, success_crop, success_location])

def test_context_awareness():
    """
    Test 2: Context-aware responses using memory
    """
    print_header("🧪 TEST 2: CONTEXT-AWARE RESPONSES")
    
    user_id = "test_user_alice"
    
    # Clean slate
    delete_user_memory(user_id)
    time.sleep(1)
    
    # Turn 1: Set context
    print_subheader("TURN 1: Set Context")
    print(f"🗣️  User: tôi tên là Alice, đang trồng cà chua ở Đà Lạt")
    print("🤖 Bot: ")
    response1, checkpoint1 = stream_chat(
        message="tôi tên là Alice, đang trồng cà chua ở Đà Lạt",
        user_id=user_id
    )
    
    # Turn 2: Pronoun reference (new thread)
    print_subheader("TURN 2: Pronoun Reference (New Thread)")
    print(f"🗣️  User: nó phù hợp với khí hậu ở đó không?")
    print("🤖 Bot: ")
    response2, checkpoint2 = stream_chat(
        message="nó phù hợp với khí hậu ở đó không?",
        user_id=user_id
    )
    
    # Validation
    print_subheader("📊 VALIDATION")
    
    # Bot should understand "nó" = cà chua, "ở đó" = Đà Lạt
    mentions_tomato = "cà chua" in response2.lower()
    mentions_dalat = "đà lạt" in response2.lower() or "dalat" in response2.lower()
    
    if mentions_tomato and mentions_dalat:
        print("✅ PASS - Bot understands context (pronouns)")
    elif mentions_tomato or mentions_dalat:
        print("⚠️  PARTIAL - Bot partially understands context")
    else:
        print("❌ FAIL - Bot doesn't understand context")
    
    return mentions_tomato and mentions_dalat

def test_user_isolation():
    """
    Test 3: Memory is isolated between different users
    """
    print_header("🧪 TEST 3: USER ISOLATION")
    
    user1 = "test_user_bob"
    user2 = "test_user_charlie"
    
    # Clean
    delete_user_memory(user1)
    delete_user_memory(user2)
    time.sleep(1)
    
    # User 1 introduces
    print_subheader("USER 1: Bob")
    print(f"🗣️  Bob: tôi tên là Bob, trồng lúa")
    print("🤖 Bot: ")
    response1, _ = stream_chat(
        message="tôi tên là Bob, trồng lúa",
        user_id=user1
    )
    
    # User 2 introduces
    print_subheader("USER 2: Charlie")
    print(f"🗣️  Charlie: tôi tên là Charlie, trồng cà phê")
    print("🤖 Bot: ")
    response2, _ = stream_chat(
        message="tôi tên là Charlie, trồng cà phê",
        user_id=user2
    )
    
    # User 1 asks who they are
    print_subheader("USER 1 MEMORY CHECK")
    print(f"🗣️  Bob: tôi là ai?")
    print("🤖 Bot: ")
    response3, _ = stream_chat(
        message="tôi là ai?",
        user_id=user1
    )
    
    # Validation
    print_subheader("📊 VALIDATION")
    
    user1_correct = "Bob" in response3 and "lúa" in response3
    user1_no_leak = "Charlie" not in response3 and "cà phê" not in response3
    
    if user1_correct and user1_no_leak:
        print("✅ PASS - Memory is properly isolated")
        print("   User 1 only sees their own data")
    else:
        print("❌ FAIL - Memory leak detected!")
        if not user1_correct:
            print("   User 1 doesn't see their own data")
        if not user1_no_leak:
            print("   User 1 sees User 2's data (SECURITY ISSUE!)")
    
    return user1_correct and user1_no_leak

def test_small_talk_memory():
    """
    Test 4: Small talk responses are personalized with memory
    """
    print_header("🧪 TEST 4: PERSONALIZED SMALL TALK")
    
    user_id = "test_user_david"
    
    # Clean
    delete_user_memory(user_id)
    time.sleep(1)
    
    # First: introduce
    print_subheader("SETUP: Introduction")
    print(f"🗣️  User: tôi tên là David")
    print("🤖 Bot: ")
    response1, _ = stream_chat(
        message="tôi tên là David",
        user_id=user_id
    )
    
    # Test: Greeting in new thread
    print_subheader("TEST: Greeting (New Thread)")
    print(f"🗣️  User: xin chào")
    print("🤖 Bot: ")
    response2, _ = stream_chat(
        message="xin chào",
        user_id=user_id
    )
    
    # Validation
    print_subheader("📊 VALIDATION")
    
    personalized = "David" in response2
    
    if personalized:
        print("✅ PASS - Greeting is personalized with name")
    else:
        print("❌ FAIL - Greeting is generic (no personalization)")
    
    return personalized

# --------------------------
# Main Test Runner
# --------------------------

def main():
    """Run all memory tests"""
    
    print_header("🚀 AGRICULTURAL AI AGENT - MEMORY TEST SUITE")
    print(f"  Server: {BASE_URL}")
    
    # Check server health
    if not check_health():
        print("\n❌ ERROR: Server is not running!")
        print("   Please start server: python app_mongodb.py")
        return
    
    print("  Status: ✅ Server is running")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    try:
        results['memory_persistence'] = test_memory_persistence()
        time.sleep(2)
        
        results['context_awareness'] = test_context_awareness()
        time.sleep(2)
        
        results['user_isolation'] = test_user_isolation()
        time.sleep(2)
        
        results['personalized_small_talk'] = test_small_talk_memory()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return
    
    # Summary
    print_header("📊 TEST SUMMARY")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\n  Tests Run: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {total - passed}")
    print()
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name.replace('_', ' ').title()}")
    
    print()
    
    if passed == total:
        print("  🎉 ALL TESTS PASSED! Memory system is working correctly!")
    elif passed > 0:
        print(f"  ⚠️  {passed}/{total} tests passed. Review failed tests.")
    else:
        print("  ❌ ALL TESTS FAILED! Memory system needs debugging.")
    
    print("=" * 70)

if __name__ == "__main__":
    main()