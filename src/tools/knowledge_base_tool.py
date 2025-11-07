from langchain_cohere import CohereRerank
from langchain_core.tools import tool
import json

class KnowledgeBaseService:
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    def _search(self, query: str) -> str:
        """Retrieve top documents from hybrid retriever, rerank, and return structured JSON with short excerpts.
        The tool returns a JSON string with a list of docs: {title, excerpt, source, doc_id}.
        """
        try:
            retrieved_docs = self.retriever.hybrid_search(query)
            reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=10)
            reranked = reranker.compress_documents(retrieved_docs, query)
            if not reranked:
                return json.dumps({"type": "text_only", "text": "Không tìm thấy thông tin phù hợp."}, ensure_ascii=False)
            docs_out = []
            for d in reranked[:8]:
                text_excerpt = (d.page_content[:600] + "...") if len(d.page_content) > 600 else d.page_content
                docs_out.append({"title": d.metadata.get("title", ""), "excerpt": text_excerpt, "source": d.metadata.get("source", ""), "id": d.metadata.get("id", "")})
            return json.dumps({"type": "docs", "query": query, "results": docs_out}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"type": "error", "message": str(e)})
    

def build_kb_tool(service: KnowledgeBaseService):
    @tool
    def search_knowledge_base(query: str) -> str:
        """Retrieve top documents from hybrid retriever, rerank, and return structured JSON with short excerpts.
        The tool returns a JSON string with a list of docs: {title, excerpt, source, doc_id}.
        """
        result = service._search(query)
        return result

    return search_knowledge_base