from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI("gemini-2.5-pro", temperature = 0)

# for m in genai.list_models():
#     print(m.name)

# def call_llm(pr`ompt: str, max_tokens: int = 400) -> str:
#     """
#     Gọi Gemini LLM để sinh phản hồi từ prompt.
#     """
#     try:
#         response = model.generate_content(
#             f"Bạn là một chuyên gia nông nghiệp, tư vấn rõ ràng, ngắn gọn.\n\nCâu hỏi: {prompt}"
#         )
#         return response.text.strip()
#     except Exception as e:
#         return f"Gemini call failed: {str(e)}"