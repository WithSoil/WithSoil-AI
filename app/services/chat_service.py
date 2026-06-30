# services/chat_service.py
import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class ChatService:
    def __init__(self):
        # LangChain 기반 Gemini 클라이언트 초기화
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7
        )

        self.llm_with_search = self.llm.bind_tools([{"google_search": {}}])

    async def chat_general(self, query: str) -> str:
        """일반 텍스트 대화"""
        response = self.llm_with_search.invoke(query)
        return response.content

    async def chat_rag(self, query: str) -> str:
        """RAG 기반 대화 (추후 Vector DB Retriever 연동부)"""
        # TODO: LangChain의 create_retrieval_chain 이나 RetrievalQA 파이프라인 추가 예정
        return "농부 일지 및 공공데이터 RAG 엔진 뼈대입니다."

    async def chat_with_image(self, query: str, image_bytes: bytes, mime_type: str) -> str:
        """이미지 분석 멀티모달 대화"""
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
            ]
        )
        response = self.llm_with_search.invoke([message])
        return response.content