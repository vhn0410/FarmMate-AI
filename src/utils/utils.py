from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage, SystemMessage


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