import duckdb
import json
import os
import hashlib
from src.interfaces.base_chunkstoredb import BaseDuckChunkStore
from langchain_core.documents import Document

class DuckDBChunkStore(BaseDuckChunkStore):
    def __init__(self, db_path="chunks.duckdb"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Create DuckDB database and FTS index if not exists."""
        conn = duckdb.connect(self.db_path)

        # Enable FTS extension (built-in)
        conn.execute("INSTALL fts;")
        conn.execute("LOAD fts;")

        # Create main table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER,
                chunk_text TEXT,
                raw_text TEXT,
                source TEXT,
                file_hash TEXT,
                metadata TEXT
            );
        """)
        
        # Create index on file_hash for quick lookup
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_hash ON chunks(file_hash);
        """)

        # Create FTS index
        try:
            conn.execute("""
                PRAGMA create_fts_index('chunks', 'id', 'chunk_text', 'raw_text',
                                       overwrite=0);
            """)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"⚠️ FTS index creation note: {e}")

        conn.close()

    def _get_file_hash(self, filepath):
        """Generate hash for file to detect duplicates."""
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except:
            return None

    def is_file_processed(self, filepath):
        """Check if file has already been processed."""
        file_hash = self._get_file_hash(filepath)
        if not file_hash:
            return False
            
        conn = duckdb.connect(self.db_path)
        result = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_hash = ?", 
            (file_hash,)
        ).fetchone()
        conn.close()
        
        return result[0] > 0 if result else False

    def save_chunks(self, chunks):
        """Save processed chunks into duckdb."""
        if not chunks:
            print("⚠️ No chunks to save")
            return
            
        conn = duckdb.connect(self.db_path)

        # Get the current max ID
        result = conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()
        current_max_id = result[0] if result else 0

        # Get file hash from first chunk's source
        first_source = chunks[0].metadata.get("source", "")
        file_hash = self._get_file_hash(first_source) if os.path.exists(first_source) else None

        # Check if file already processed
        if file_hash:
            existing = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_hash = ?", 
                (file_hash,)
            ).fetchone()
            
            if existing and existing[0] > 0:
                conn.close()
                print(f"⚠️ File already processed: {first_source}")
                print(f"   Skipping {len(chunks)} chunks to avoid duplicates")
                return

        saved_count = 0
        for idx, doc in enumerate(chunks, start=1):
            chunk_text = doc.page_content
            raw_text = doc.metadata.get("original_content", "")
            source = doc.metadata.get("source", "unknown")
            metadata = json.dumps(doc.metadata, ensure_ascii=False)

            # Manually assign ID
            new_id = current_max_id + idx

            conn.execute("""
                INSERT INTO chunks (id, chunk_text, raw_text, source, file_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (new_id, chunk_text, raw_text, source, file_hash, metadata))
            
            saved_count += 1

        conn.close()
        print(f"✅ Saved {saved_count} new chunks into DuckDB ({self.db_path})")

    def delete_chunks_by_source(self, source_path):
        """Delete all chunks from a specific source file."""
        file_hash = self._get_file_hash(source_path)
        if not file_hash:
            print(f"⚠️ Cannot calculate hash for: {source_path}")
            return 0
            
        conn = duckdb.connect(self.db_path)
        result = conn.execute(
            "DELETE FROM chunks WHERE file_hash = ? RETURNING id", 
            (file_hash,)
        ).fetchall()
        conn.close()
        
        deleted_count = len(result)
        print(f"🗑️ Deleted {deleted_count} chunks from: {source_path}")
        return deleted_count

    def get_all_sources(self):
        """Get list of all processed source files."""
        conn = duckdb.connect(self.db_path)
        result = conn.execute("""
            SELECT DISTINCT source, COUNT(*) as chunk_count 
            FROM chunks 
            GROUP BY source
            ORDER BY source
        """).fetchall()
        conn.close()
        return result

    def get_all_documents(self):
        """Return ALL documents from DuckDB as List[Document]."""
        conn = duckdb.connect(self.db_path)

        sql = """
            SELECT id, chunk_text, metadata
            FROM chunks
            ORDER BY id ASC;
        """

        rows = conn.execute(sql).fetchall()
        conn.close()

        documents = []

        for r in rows:
            doc_id = r[0]
            chunk_text = r[1]
            metadata_json = json.loads(r[2]) if r[2] else {}

            # Add chunk_id if missing
            metadata_json["chunk_id"] = doc_id

            raw_text = metadata_json.get("original_content", "")

            doc = Document(
                page_content=chunk_text,
                metadata={
                    **metadata_json,
                    "raw_text": raw_text
                }
            )

            documents.append(doc)

        return documents
    


"""
# Xem các file đã xử lý
sources = chunk_store.get_all_sources()
for source, count in sources:
    print(f"{source}: {count} chunks")

# Xóa chunks của một file cụ thể (nếu muốn reprocess)
chunk_store.delete_chunks_by_source("path/to/old_file.pdf")

# Sau đó có thể xử lý lại file đó

"""