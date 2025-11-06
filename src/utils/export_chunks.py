import json 
from langchain_core.documents import Document


def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data

def export_chunk_json_to_document_langchain(filepath="chunks_export.json"):
    """Convert json data chunk exported to Document langchain format"""

    # === Step 1: Load JSON file ===
    with open(filepath, "r", encoding="utf-8") as f:
        export_data = json.load(f)

    documents = [
        Document(
            page_content=item["enhanced_content"],
            metadata={
                "chunk_id": item.get("chunk_id"),
                "source": "so_tay_lua_chat_luong_cao.pdf",
                "raw_text": item["metadata"]["original_content"].get("raw_text", "")
            }
        )
        for item in export_data
    ]

    return documents