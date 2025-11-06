
from services.load_vector_store import load_vector_store
import os
# After your retrieval
import json
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from services.export_chunks_to_json import export_chunk_json_to_document_langchain
from services.ingesion_pipeline import run_complete_ingestion_pipeline
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank
# database = load_vector_store()
persist_directory = "db_agriculture2/chroma_db"
pdf_dir = "./docs"
load_dotenv()
if os.path.exists(persist_directory):
    db = load_vector_store(persist_directory)
else:
    db = None
print("Run complete ingestion pipeline")
new_db = run_complete_ingestion_pipeline(pdf_dir)

if new_db:
    db = new_db

# retriever = db.as_retriever(search_kwargs={"k": 3})
# query = "Hãy trình bày quy trình gồm hướng dẫn gieo sạ theo hàng hoặc theo cụm bằng máy ở ĐBSCL"
# chunks = retriever.invoke(query)
# print(chunks)


# print(documents)
# Query the vector store
# print("Quy trình gồm hướng dẫn gieo sạ theo hàng hoặc theo cụm bằng máy ở ĐBSCL")
# query = "How many components in Figure 1: The Transformer - model architecture? and explain it clear"



# 1. Vector Retriever (Semantic Search/Dense Retrieval)
vector_retriever = db.as_retriever(search_kwargs={"k": 15})

# 2. BM25 Retriever (Keyword Search/Sparse Retrieval)

documents = export_chunk_json_to_document_langchain("db_agriculture2/chunks_export.json")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15

# 3. hybrid retriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

print("Setup complete!\n")
print("STEP 1: Hybrid Search Results")
print("-"*50)
query = "Đối với vụ Đông Xuân trên đất phù sa, lượng phân đa lượng (N, P2O5, K2O) khuyến nghị là bao nhiêu kg/ha?"
retrieved_docs = hybrid_retriever.invoke(query)  # Get top 25 for reranking

# for i, doc in enumerate(retrieved_docs, 1):
#     print(f"{i:2d}. {doc.page_content}")

# print(f"\n(Retrieved {len(retrieved_docs)} total chunks for reranking)\n")
#=============================================================================================
print("STEP 2: After Cohere Reranking (Top 10)")
print("-"*50)

# Initialize Cohere reranker
reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=10)

# Rerank the retrieved documents
reranked_docs = reranker.compress_documents(retrieved_docs, query)

# Show reranked results
for i, doc in enumerate(reranked_docs, 1):
    print(f"{i:2d}. {doc.page_content}")

print("\n" + "="*80)
print("ANALYSIS:")
print("✅ Hybrid Search: Mixed relevant and irrelevant results")
print("✅ Reranking: Most relevant Tesla financial/production info at top")
print("✅ Notice how reranking moved the most contextually relevant chunks higher")

# Optional: Show the difference more clearly
# print("\n" + "="*80)
# print("KEY IMPROVEMENTS AFTER RERANKING:")
# print("-"*40)


# hybrid_top_5 = [doc.page_content for doc in retrieved_docs[:5]]
# reranked_top_5= [doc.page_content for doc in reranked_docs[:5]]

# print("BEFORE (Hybrid Top 5):")
# for i, content in enumerate(hybrid_top_5, 1):
#     print(f"  {i}. {content}")

# print("\nAFTER (Reranked Top 5):")
# for i, content in enumerate(reranked_top_5, 1):
#     print(f"  {i}. {content}")


def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    try:
        # Initialize LLM (needs vision model for images)
        # llm = ChatOpenAI(model="gpt-4o", temperature=0)
        llm = ChatGoogleGenerativeAI(
            # model="gemini-2.5-pro",  # Miễn phí, rất nhanh
            model="gemini-2.5-flash",  # Miễn phí, rất nhanh
            temperature=0
        )
        
        # Build the text prompt
        prompt_text = f"""Dựa trên những tài liệu được cung cấp, bạn hãy trả lời câu hỏi sau: {query}

# Nội dung cần được phân tích:
# """
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Tài liệu {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"Văn bản:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "Bảng:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Bảng {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Vui lòng đưa ra một câu trả lời rõ ràng và toàn diện, sử dụng văn bản, bảng và hình ảnh ở phía trên. Nếu các tài liệu không chứa đủ thông tin để trả lời câu hỏi, hãy nói: "Tôi không có đủ thông tin để trả lời câu hỏi đó dựa trên các tài liệu được cung cấp.
TRẢ LỜI:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

# # Usage
final_answer = generate_final_answer(reranked_docs, query)
print(final_answer)

# RAG tài liệu tiếng việt
# Tạo workflow đơn giản để lấy dữ liệu
# Có thể cải thiện sau đó

