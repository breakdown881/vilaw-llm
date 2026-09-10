"""
Pipeline Sinh Dữ liệu Tổng hợp DPO Đa Nhà Cung Cấp (Multi-Provider: Gemini / Groq / OpenAI)
Dự án: VILaw-LLM
File: data/scripts/generate_synthetic_data.py

Hỗ trợ MIỄN PHÍ 100%:
1. Google Gemini API (gemini-1.5-flash / gemini-2.0-flash qua OpenAI-compatible endpoint)
2. Groq Cloud API (llama-3.3-70b-versatile / qwen-2.5-32b)
3. OpenAI API (gpt-4o-mini)
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DPO_FILE = PROCESSED_DIR / "legal_dpo_pairs.parquet"

# ==============================================================================
# KHỞI TẠO CLIENT LINH HOẠT THEO NHÀ CUNG CẤP (PROVIDER)
# ==============================================================================
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

if PROVIDER == "gemini":
    API_KEY = os.getenv("GEMINI_API_KEY")
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    logger.info("🔧 Sử dụng Provider: Google Gemini API (Miễn phí)")
elif PROVIDER == "groq":
    API_KEY = os.getenv("GROQ_API_KEY")
    BASE_URL = "https://api.groq.com/openai/v1"
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    logger.info("🔧 Sử dụng Provider: Groq Cloud API (Miễn phí)")
else:
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = None
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    logger.info("🔧 Sử dụng Provider: OpenAI API")

if not API_KEY:
    logger.warning(
        f"⚠️ Chưa tìm thấy API Key cho provider '{PROVIDER}'!\n"
        f"   - Nếu dùng Gemini: Thêm GEMINI_API_KEY=xxx vào file .env\n"
        f"   - Nếu dùng Groq: Thêm GROQ_API_KEY=xxx vào file .env\n"
        f"   - Hoặc chỉnh LLM_PROVIDER trong .env"
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None

SYSTEM_PROMPT = """Bạn là một Chuyên gia Pháp lý cao cấp và Giám khảo Đánh giá LLM.
Nhiệm vụ của bạn là nhận vào một câu hỏi pháp lý và tạo ra 2 phiên bản câu trả lời bằng TIẾNG VIỆT để huấn luyện mô hình DPO:

1. CHOSEN (Câu trả lời mẫu mực):
   - Nêu rõ căn cứ pháp lý: Trích dẫn chính xác Điều, Khoản, tên văn bản luật hiện hành của Việt Nam.
   - Lập luận logic: Phân tích áp dụng điều luật vào tình huống thực tế của câu hỏi.
   - Khuyến nghị/Lưu ý: Hướng dẫn người hỏi các bước thực hiện an toàn về mặt pháp lý.

2. REJECTED (Câu trả lời kém chất lượng):
   - Trả lời chung chung, né tránh, phán đoán cảm tính ("theo tôi nghĩ", "thường thì...").
   - KHÔNG trích dẫn bất kỳ Điều hay Khoản luật cụ thể nào.
   - Thiếu khuyến nghị hành động hoặc lời khuyên mơ hồ, có thể gây rủi ro pháp lý.

Format bắt buộc: Bạn PHẢI trả về duy nhất một chuỗi JSON hợp lệ với đúng 2 key: "chosen" và "rejected".
"""

def generate_dpo_pair(question: str, reference_answer: str = "") -> Optional[Dict[str, str]]:
    if not client:
        raise ValueError("LLM Client chưa được khởi tạo. Vui lòng kiểm tra API Key trong file .env!")

    user_content = f"Câu hỏi pháp lý: {question}\n"
    if reference_answer:
        user_content += f"Tài liệu tham khảo/Gợi ý nội dung: {reference_answer[:500]}\n"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=3000
        )

        content = response.choices[0].message.content.strip()
        
        # Làm sạch nếu model bọc trong markdown ```json ... ```
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)

        return {
            "prompt": question,
            "chosen": data.get("chosen", "").strip(),
            "rejected": data.get("rejected", "").strip()
        }

    except Exception as e:
        logger.error(f"❌ Lỗi khi sinh dữ liệu: {str(e)}")
        return None


def run_synthetic_pipeline(num_samples: int = 20, batch_save_interval: int = 5):
    input_file = PROCESSED_DIR / "legal_sft_train.parquet"
    if not input_file.exists():
        logger.error(f"❌ Không tìm thấy file {input_file}!")
        return

    df_sft = pd.read_parquet(input_file)
    logger.info(f"Đã nạp {len(df_sft):,} mẫu từ SFT train set.")

    existing_records = []
    processed_prompts = set()

    if OUTPUT_DPO_FILE.exists():
        df_existing = pd.read_parquet(OUTPUT_DPO_FILE)
        existing_records = df_existing.to_dict(orient="records")
        processed_prompts = set(df_existing["prompt"].str.strip().tolist())
        logger.info(f"-> Tìm thấy checkpoint cũ: Đã có {len(existing_records):,} mẫu DPO.")

    candidates = df_sft[~df_sft["question"].str.strip().isin(processed_prompts)]
    sample_targets = candidates.head(num_samples)
    logger.info(f"🚀 Bắt đầu sinh {len(sample_targets)} mẫu DPO mới với model {MODEL_NAME}...")

    new_records = []
    count = 0

    for idx, row in sample_targets.iterrows():
        q = row["question"]
        ref_a = row.get("answer", "")

        logger.info(f"[{count+1}/{len(sample_targets)}] Đang xử lý: {q[:60]}...")
        pair = generate_dpo_pair(question=q, reference_answer=ref_a)

        if pair and pair["chosen"] and pair["rejected"]:
            new_records.append(pair)
            existing_records.append(pair)
            count += 1
        else:
            logger.warning(f"⚠️ Bỏ qua mẫu do không sinh được cặp hợp lệ.")

        time.sleep(2.5)  # Delay 2.5s tránh chạm trần TPM/RPM rate limit của Groq

        if len(new_records) % batch_save_interval == 0 and len(new_records) > 0:
            pd.DataFrame(existing_records).to_parquet(OUTPUT_DPO_FILE, index=False)
            logger.info(f"💾 Checkpoint: Đã lưu {len(existing_records)} mẫu vào {OUTPUT_DPO_FILE.name}")

    if new_records:
        pd.DataFrame(existing_records).to_parquet(OUTPUT_DPO_FILE, index=False)
        logger.info(f"✅ HOÀN TẤT! Đã lưu tổng cộng {len(existing_records):,} mẫu DPO tại {OUTPUT_DPO_FILE}")


def main():
    logger.info("========== BẮT ĐẦU PIPELINE SYNTHETIC DATA GENERATION ==========")
    run_synthetic_pipeline(num_samples=20, batch_save_interval=5)


if __name__ == "__main__":
    main()
