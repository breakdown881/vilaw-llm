# ==============================================================================
# DOCKERFILE CHO VILAW-LLM FASTAPI SERVICE
# Base Image: Python 3.10 Slim (Tối ưu dung lượng nhẹ)
# ==============================================================================

FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# TODO 1: Copy file requirements.txt vào thư mục /app và cài đặt thư viện bằng pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# TODO 2: Copy toàn bộ mã nguồn src/ và data/ vào container
COPY src/ ./src/


# TODO 3: Mở cổng (EXPOSE) 8000 cho FastAPI
EXPOSE 8000


# TODO 4: Lệnh CMD khởi chạy uvicorn server lắng nghe tại 0.0.0.0:8000
CMD ["uvicorn", "src.serving.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
