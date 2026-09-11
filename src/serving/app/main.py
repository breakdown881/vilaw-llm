"""
FastAPI Server Khởi Chạy Dịch Vụ VILaw-LLM
Dự án: VILaw-LLM
File: src/serving/app/main.py

Khởi chạy máy chủ:
  uvicorn src.serving.app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.serving.app.api.legal_chat import router as legal_chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="VILaw-LLM API Service",
    description="Hệ thống API Tư vấn Pháp lý Việt Nam thông minh với mô hình VILaw-LLM (SFT + DPO)",
    version="1.0.0"
)

# Cấu hình CORS để frontend (React, Vue, Web) có thể gọi trực tiếp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký Router
app.include_router(legal_chat_router)


@app.get("/health", tags=["Health Check"])
async def health_check():
    """Endpoint kiểm tra trạng thái sống của dịch vụ"""
    return {
        "status": "healthy",
        "service": "VILaw-LLM Serving Engine",
        "version": "1.0.0",
        "supported_models": ["vilaw-sft-lora", "vilaw-dpo-lora"]
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Đang khởi chạy VILaw-LLM Server tại http://0.0.0.0:8000 ...")
    uvicorn.run("src.serving.app.main:app", host="0.0.0.0", port=8000, reload=True)
