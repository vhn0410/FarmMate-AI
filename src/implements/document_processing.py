
from src.interfaces.base_documents import BaseDocument
import json
from typing import List

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

class DocumentProcessing(BaseDocument):
    
    DocumentPath = "/docs"

    # Step 1: Partitioning PDF into atomic elements
    def partition_document(self, file_path: str = DocumentPath):
        """Extract elements from PDF using unstructured"""
        print(f"📄 Partitioning document: {file_path}")
        
        elements = partition_pdf(
            filename=file_path,  # Path to your PDF file
            strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
            infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"], # Grab images found in the PDF
            extract_image_block_to_payload=True, # Store images as base64 data you can actually use,
            languages=["vie", "en"],
            ocr_languages=["vie", "en"]
        )
        
        print(f"✅ Extracted {len(elements)} elements")
        return elements
   
    # Step 2: Chunking by title
    def create_chunks_by_title(self, elements):
        """Create intelligent chunks using title-based strategy"""
        print("🔨 Creating smart chunks...")
        
        chunks = chunk_by_title(
            elements, # The parsed PDF elements from previous step
            max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
            new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
            combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors,
            
        )
        
        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    # Step 3: Summarising chunks and converting it into LangChain documents
    def separate_content_types(self, chunk):
        """Analyze what types of content are in a chunk"""
        content_data = {
            'text': chunk.text,
            'tables': [],
            'images': [],
            'types': ['text']
        }
        
        # Check for tables and images in original elements
        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
            for element in chunk.metadata.orig_elements:
                element_type = type(element).__name__
                
                # Handle tables
                if element_type == 'Table':
                    content_data['types'].append('table')
                    table_html = getattr(element.metadata, 'text_as_html', element.text)
                    content_data['tables'].append(table_html)
                
                # Handle images
                elif element_type == 'Image':
                    if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                        content_data['types'].append('image')
                        content_data['images'].append(element.metadata.image_base64)
        
        content_data['types'] = list(set(content_data['types']))
        return content_data


    def create_ai_enhanced_summary(self, text: str, tables: List[str], images: List[str]) -> str:
        """Create AI-enhanced summary for mixed content"""

        try:
            # Initialize LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0
            )

            # --- Build main prompt ---
            prompt_text = f"""
    Bạn là hệ thống tạo mô tả phục vụ cho việc truy xuất trong RAG.

    QUY TẮC TUYỆT ĐỐI:
    - Không được bắt đầu câu trả lời bằng các cụm như: "Chắc chắn rồi", "Dưới đây là", "Tôi sẽ", "Vâng", "Được thôi".
    - Không được viết lời mở đầu, lời chào, lời giải thích về quy trình.
    - Không được viết câu meta như "Theo yêu cầu của bạn".
    - Chỉ xuất *nội dung mô tả thuần túy*.

    NỘI DUNG CẦN PHÂN TÍCH

    VĂN BẢN:
    {text}

    """

            # ✅ Add tables
            if tables:
                prompt_text += "\nBẢNG DỮ LIỆU:\n"
                for i, table in enumerate(tables):
                    prompt_text += f"- Bảng {i+1}:\n{table}\n"

            # ✅ Add instructions
            prompt_text += """
    NHIỆM VỤ:
    Hãy tạo một mô tả có thể tìm kiếm, bao gồm:

    1. Các dữ kiện quan trọng, số liệu, con số  
    2. Các chủ đề và khái niệm chính  
    3. Những câu hỏi mà nội dung có thể trả lời  
    4. Phân tích các hình ảnh (nếu có)  
    5. Các từ khóa tìm kiếm và từ khóa thay thế  

    Lưu ý: Chỉ xuất nội dung mô tả. Không được viết lời dẫn.

    BẮT ĐẦU MÔ TẢ:
    """

            # --- Build message payload ---
            message_content = [
                {"type": "text", "text": prompt_text}
            ]

            # ✅ Add images (base64)
            for image_base64 in images:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })

            # Send
            message = HumanMessage(content=message_content)
            response = llm.invoke([message])

            return response.content

        except Exception as e:
            print(f"❌ AI summary failed: {e}")
            summary = f"{text[:300]}..."
            if tables:
                summary += f" [Contains {len(tables)} tables]"
            if images:
                summary += f" [Contains {len(images)} images]"
            return summary


    

    def summarise_chunks(self, chunks):
        """Process all chunks with AI Summaries"""
        print("🧠 Processing chunks with AI Summaries...")
        
        langchain_documents = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            current_chunk = i + 1
            print(f"   Processing chunk {current_chunk}/{total_chunks}")
            
            # Analyze chunk content
            content_data = self.separate_content_types(chunk)
            
            # Debug prints
            print(f"     Types found: {content_data['types']}")
            print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
            
            # Create AI-enhanced summary if chunk has tables/images
            if content_data['tables'] or content_data['images']:
                print(f"     → Creating AI summary for mixed content...")
                try:
                    enhanced_content = self.create_ai_enhanced_summary(
                        content_data['text'],
                        content_data['tables'], 
                        content_data['images']
                    )
                    print(f"     → AI summary created successfully")
                    print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
                except Exception as e:
                    print(f"     ❌ AI summary failed: {e}")
                    enhanced_content = content_data['text']
            else:
                print(f"     → Using raw text (no tables/images)")
                enhanced_content = content_data['text']
            
            # Create LangChain Document with rich metadata
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "original_content": json.dumps({
                        "raw_text": content_data['text'],
                        "tables_html": content_data['tables'],
                        "images_base64": content_data['images']
                    })
                }
            )
            
            langchain_documents.append(doc)
        
        print(f"✅ Processed {len(langchain_documents)} chunks")
        return langchain_documents
    



   