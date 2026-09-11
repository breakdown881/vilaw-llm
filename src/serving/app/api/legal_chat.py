"""
API Router Phục Vụ Chat Pháp Lý với Hỗ Trợ Server-Sent Events (SSE Streaming)
Dự án: VILaw-LLM
File: src/serving/app/api/legal_chat.py

HƯỚNG DẪN THỰC HÀNH CHO HỌC VIÊN:
File này là khung sườn (skeleton) chuẩn FastAPI Streaming.
Nhiệm vụ của bạn: Hoàn thiện các khối # TODO theo hướng dẫn của Mentor.
"""

import os
import json
import asyncio
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import tầng Guardrails từ core
from src.serving.app.core.guardrails import validate_input_safety, apply_output_guardrail

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Legal Chat"])


# ==============================================================================
# BƯỚC 1: ĐỊNH NGHĨA PYDANTIC SCHEMAS (CHUẨN OPENAI COMPATIBLE)
# ==============================================================================
class ChatMessage(BaseModel):
    role: str = Field(..., description="Vai trò: 'user', 'assistant' hoặc 'system'")
    content: str = Field(..., description="Nội dung tin nhắn")


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Lịch sử hội thoại")
    model: str = Field(default="vilaw-llm-dpo", description="Tên mô hình")
    stream: bool = Field(default=True, description="Bật/tắt chế độ streaming SSE")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1024, le=2048)


# ==============================================================================
# BƯỚC 2: HÀM STREAM GENERATOR (SSE: text/event-stream)
# ==============================================================================
async def legal_stream_generator(user_prompt: str):
    """
    Generator tạo luồng Server-Sent Events (SSE):
    - Kiểm tra Guardrails: Nếu vi phạm, stream lý do từ chối và ngắt ngay.
    - Nếu an toàn: Mô phỏng sinh stream từng token và tự động gắn Disclaimer ở cuối.
    """
    # 1. Kiểm tra Guardrail đầu vào
    is_safe, error_reason = validate_input_safety(user_prompt)
    if not is_safe:
        # TODO 1: Bắn ra lý do vi phạm theo định dạng SSE và kết thúc stream
        payload = json.dumps({"content": error_reason}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 2. Tạo nội dung phản hồi mẫu (hoặc gọi qua Model Engine sau này)
    sample_legal_response = (
        "Căn cứ theo quy định của pháp luật Việt Nam hiện hành, "
        "yêu cầu của bạn được xem xét và xử lý theo các bước quy định cụ thể.\n"
        "1. Về cơ sở pháp lý: Bạn cần đối chiếu với các điều khoản liên quan trong văn bản luật chuyên ngành.\n"
        "2. Về quyền và nghĩa vụ: Các bên có trách nhiệm tuân thủ đúng nội dung đã cam kết."
    )
    
    # Gắn Tuyên bố miễn trừ trách nhiệm pháp lý qua hàm apply_output_guardrail
    final_response = apply_output_guardrail(sample_legal_response)

    # TODO 2: Tách final_response thành từng cụm từ/từ (token) và yield từng phần tử
    words = final_response.split(" ")
    for word in words:
        chunk = word + " "
        payload = json.dumps({"content": chunk}, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        await asyncio.sleep(0.04) # Tạo độ trễ tự nhiên giữa các từ

    # TODO 3: Gửi tín hiệu báo hiệu kết thúc luồng stream
    yield "data: [DONE]\n\n"


# ==============================================================================
# BƯỚC 3: ENDPOINT TIẾP NHẬN YÊU CẦU /chat/completions
# ==============================================================================
@router.post("/chat/completions")
async def chat_endpoint(request: ChatCompletionRequest):
    """
    Endpoint chính tiếp nhận yêu cầu hỏi đáp pháp lý:
    Hỗ trợ cả Streaming (SSE) và Non-streaming.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Danh sách messages không được để trống.")

    # Lấy câu hỏi cuối cùng của người dùng (role == 'user')
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="Không tìm thấy tin nhắn từ người dùng (role 'user').")
    
    latest_user_prompt = user_messages[-1].content

    # TODO 4: Kiểm tra request.stream
    # Nếu request.stream == True:
    #   Trả về StreamingResponse(legal_stream_generator(latest_user_prompt), media_type="text/event-stream")
    # Ngược lại:
    #   Kiểm tra validate_input_safety(latest_user_prompt), nếu an toàn trả về dict JSON chuẩn OpenAI
    if request.stream:
        return StreamingResponse(
            legal_stream_generator(latest_user_prompt),
            media_type="text/event-stream"
        )
    else:
        is_safe, error_reason = validate_input_safety(latest_user_prompt)
        if not is_safe:
            return {"role": "assistant", "content": error_reason}
        
        reply = apply_output_guardrail("Căn cứ các quy định pháp luật hiện hành...")
        return {
            "model": request.model,
            "choices": [{
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop"
            }]
        }
