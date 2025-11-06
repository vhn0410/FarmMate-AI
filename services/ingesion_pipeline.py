import os
import glob
import hashlib
import json
from services.documents_processing import partition_document, create_chunks_by_title, summarise_chunks
from services.embedding_chunks_vectorstore import create_vector_store
from services.export_chunks_to_json import export_chunks_to_json


def get_file_hash(file_path):
    """Tạo hash MD5 cho file để phát hiện thay đổi nội dung."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def run_complete_ingestion_pipeline(pdf_dir: str, meta_path="db_agriculture2/processed_files.json"):
    """Chỉ xử lý file PDF mới hoặc thay đổi."""
    print("🚀 Starting Incremental RAG Ingestion Pipeline")
    print("=" * 50)

    # Load metadata cũ
    processed_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            processed_meta = json.load(f)

    # Tìm tất cả file PDF
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    new_files = []

    for pdf_path in pdf_files:
        file_hash = get_file_hash(pdf_path)
        if pdf_path not in processed_meta or processed_meta[pdf_path] != file_hash:
            print(f"📄 Mới hoặc thay đổi: {pdf_path}")
            new_files.append((pdf_path, file_hash))
        else:
            print(f"✅ Bỏ qua (đã có): {pdf_path}")

    if not new_files:
        print("✅ Không có file mới. Bỏ qua ingestion.")
        return None

    all_summarised_chunks = []

    for pdf_path, file_hash in new_files:
        elements = partition_document(pdf_path)
        chunks = create_chunks_by_title(elements)
        summarised_chunks = summarise_chunks(chunks)
        all_summarised_chunks.extend(summarised_chunks)
        processed_meta[pdf_path] = file_hash

    export_chunks_to_json(all_summarised_chunks)
    db = create_vector_store(all_summarised_chunks, persist_directory="db_agriculture2/chroma_db")

    # Lưu metadata mới
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(processed_meta, f, ensure_ascii=False, indent=2)

    print("🎉 Incremental ingestion completed successfully!")
    return db
