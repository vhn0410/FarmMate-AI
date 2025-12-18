from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, SystemMessage
import os

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


def load_prompt(file_name: str, **kwargs) -> str:
    # 1. Lấy đường dẫn thư mục chứa file utils.py (đang là .../src/utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Đi ngược lên 1 cấp để vào thư mục src/
    src_dir = os.path.dirname(current_dir)
    
    # 3. Kết nối vào thư mục prompts/ nằm trong src/
    path = os.path.join(src_dir, "prompts", file_name)
    
    # Kiểm tra lỗi để dễ debug
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file prompt tại: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 4. Fill các biến như {name} vào nội dung prompt
    return content.format(**kwargs)
