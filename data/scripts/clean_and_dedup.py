"""
Pipeline Làm sạch & Chuẩn hoá Dữ liệu Pháp lý (Data Cleaning & Deduplication)
Dự án: VILaw-LLM
File: data/scripts/clean_and_dedup.py

Mục tiêu:
1. Đọc dữ liệu thô từ data/raw/
2. Chuẩn hoá Unicode NFC và dọn dẹp khoảng trắng
3. Bung mảng (flatten) dữ liệu hỏi-đáp lồng nhau
4. Lọc bỏ dữ liệu rác, quá ngắn, không có cơ sở pháp lý
5. Loại bỏ dữ liệu trùng lặp (Deduplication)
6. Chia tập Train (90%) và Val (10%), lưu ra data/processed/*.parquet
"""

import os
import re
import sys
import unicodedata
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# BƯỚC 1: HÀM CHUẨN HOÁ VĂN BẢN (TEXT NORMALIZATION)
# ==============================================================================
def normalize_text(text: str) -> str:
    """
    Chuẩn hoá văn bản Tiếng Việt:
    - Chuyển về dạng Unicode NFC (dựng sẵn)
    - Xoá khoảng trắng thừa, tab, xuống dòng liên tiếp
    """
    if not isinstance(text, str):
        return ""
    
    # TODO 1: Áp dụng unicodedata.normalize để đưa text về chuẩn 'NFC'
    text = unicodedata.normalize('NFC', text)
    
    # TODO 2: Dùng regex re.sub để thay thế nhiều khoảng trắng/xuống dòng liên tiếp thành 1 khoảng trắng duy nhất
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ==============================================================================
# BƯỚC 2: TRÍCH XUẤT VÀ BUNG MẢNG (DATA EXTRACTION & FLATTENING)
# ==============================================================================
def process_vietnamese_legal_qa(file_path: Path) -> list[dict]:
    """
    Xử lý file vietnamese_legal_qa_raw.parquet:
    - Mỗi dòng có trường 'generated_qa_pairs' là 1 danh sách/mảng các dict: {'question': ..., 'answer': ...}
    - Cần duyệt qua từng dòng và lấy hết các cặp QA này ra.
    """
    logger.info(f"Đang đọc và bung mảng dữ liệu từ: {file_path.name}")
    df = pd.read_parquet(file_path)
    records = []

    for idx, row in df.iterrows():
        qa_pairs = row.get("generated_qa_pairs")
        doc_name = row.get("doc_name", "")

        # TODO 3: Kiểm tra nếu qa_pairs là danh sách hoặc numpy array (iterable)
        # Duyệt qua từng item trong qa_pairs, trích xuất 'question' và 'answer'
        # Sau đó chuẩn hoá qua hàm normalize_text()
        # Thêm dict vào records: {"question": q, "answer": a, "source": f"legal_qa_{doc_name}"}
        
        if qa_pairs is not None and len(qa_pairs) > 0:
            for pair in qa_pairs:
                if isinstance(pair, dict):
                    q = normalize_text(pair.get("question", ""))
                    a = normalize_text(pair.get("answer", ""))
                    if q and a:
                        records.append({
                            "question": q,
                            "answer": a,
                            "source": f"legal_qa_{doc_name}"
                        })

    logger.info(f"✓ Đã trích xuất {len(records):,} cặp QA từ {file_path.name}")
    return records


def process_vibid_lqa(file_path: Path) -> list[dict]:
    """
    Xử lý file vibid_lqa_raw.parquet:
    - Các cột: ['context', 'question', 'answer']
    - Lưu ý: Nhiều câu trả lời chỉ ghi tên luật, nếu muốn chất lượng cao có thể ghép context nếu cần.
    """
    logger.info(f"Đang đọc dữ liệu từ: {file_path.name}")
    df = pd.read_parquet(file_path)
    records = []

    for idx, row in df.iterrows():
        # TODO 4: Lấy question và answer từ row, chuẩn hoá bằng normalize_text()
        # Thêm vào records: {"question": q, "answer": a, "source": "vibid_lqa"}
        q = normalize_text(row.get("question", ""))
        a = normalize_text(row.get("answer", ""))
        if q and a:
            records.append({
                "question": q,
                "answer": a,
                "source": "vibid_lqa"
            })

    logger.info(f"✓ Đã trích xuất {len(records):,} cặp QA từ {file_path.name}")
    return records


# ==============================================================================
# BƯỚC 3: BỘ LỌC CHẤT LƯỢNG PHÁP LÝ (QUALITY FILTERING)
# ==============================================================================
def filter_quality(records: list[dict]) -> list[dict]:
    """
    Loại bỏ các mẫu không đạt chuẩn chất lượng:
    - Câu hỏi quá ngắn (< 15 ký tự)
    - Câu trả lời quá ngắn (< 30 ký tự)
    - Câu trả lời KHÔNG chứa từ khoá cơ sở pháp lý (điều, khoản, luật, nghị định, thông tư, quy định)
    """
    logger.info("Đang áp dụng bộ lọc chất lượng pháp lý (Quality Filter)...")
    filtered = []
    
    legal_keywords = [
        "điều", "khoản", "điểm", "luật", "nghị định", 
        "thông tư", "quy định", "bộ luật", "pháp lệnh", "quy chuẩn"
    ]

    for item in records:
        q = item.get("question", "")
        a = item.get("answer", "")
        
        # TODO 5: Viết điều kiện lọc:
        # - q phải dài hơn 15 ký tự: len(q) >= 15
        # - a phải dài hơn 30 ký tự: len(a) >= 30
        # - a phải chứa ít nhất 1 từ khoá trong legal_keywords (gợi ý: any(kw in a.lower() for kw in legal_keywords))
        # Nếu thoả mãn thì filtered.append(item)
        
        has_keyword = any(kw in a.lower() for kw in legal_keywords)
        if len(q) >= 15 and len(a) >= 30 and has_keyword:
            filtered.append(item)

    logger.info(f"✓ Dữ liệu sau lọc chất lượng: {len(filtered):,} / {len(records):,} mẫu")
    return filtered


# ==============================================================================
# BƯỚC 4: LOẠI BỎ TRÙNG LẶP (DEDUPLICATION)
# ==============================================================================
def deduplicate_records(records: list[dict]) -> list[dict]:
    """
    Loại bỏ các câu hỏi bị lặp lại hoàn toàn (Exact Dedup theo question)
    """
    logger.info("Đang kiểm tra và loại bỏ trùng lặp (Deduplication)...")
    seen_questions = set()
    unique_records = []

    for item in records:
        # TODO 6: Kiểm tra nếu question đã tồn tại trong set seen_questions hay chưa
        # Nếu chưa có: thêm vào set và đưa item vào unique_records
        q_key = item.get("question", "").strip().lower()
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            unique_records.append(item)

    removed_count = len(records) - len(unique_records)
    logger.info(f"✓ Đã loại bỏ {removed_count:,} mẫu trùng lặp. Còn lại: {len(unique_records):,} mẫu độc bản")
    return unique_records


# ==============================================================================
# BƯỚC 5: CHIA TẬP DỮ LIỆU VÀ LƯU PARQUET (TRAIN / VAL SPLIT)
# ==============================================================================
def split_and_save(records: list[dict], train_ratio: float = 0.9):
    """
    Chia dữ liệu theo tỉ lệ 90% Train / 10% Val và lưu ra Parquet
    """
    df = pd.DataFrame(records)
    logger.info(f"Tổng số mẫu đưa vào chia Train/Val: {len(df):,}")

    train_df, val_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=42,
        shuffle=True
    )

    train_path = PROCESSED_DIR / "legal_sft_train.parquet"
    val_path = PROCESSED_DIR / "legal_sft_val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    logger.info(f"✓ Đã lưu tập Train ({len(train_df):,} mẫu) tại: {train_path}")
    logger.info(f"✓ Đã lưu tập Val ({len(val_df):,} mẫu) tại: {val_path}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    logger.info("========== BẮT ĐẦU PIPELINE LÀM SẠCH DỮ LIỆU ==========")
    
    file_legal_qa = RAW_DIR / "vietnamese_legal_qa_raw.parquet"
    file_vibid = RAW_DIR / "vibid_lqa_raw.parquet"

    if not file_legal_qa.exists() or not file_vibid.exists():
        logger.error("❌ Không tìm thấy file dữ liệu thô trong data/raw/! Hãy chạy download_datasets.py trước.")
        sys.exit(1)

    # 1. Trích xuất
    records_legal = process_vietnamese_legal_qa(file_legal_qa)
    records_vibid = process_vibid_lqa(file_vibid)
    all_records = records_legal + records_vibid
    logger.info(f"-> Tổng số mẫu thô gom được: {len(all_records):,}")

    # 2. Lọc chất lượng
    clean_records = filter_quality(all_records)

    # 3. Loại trùng
    dedup_records = deduplicate_records(clean_records)

    # 4. Chia và lưu
    split_and_save(dedup_records, train_ratio=0.9)

    logger.info("========== HOÀN TẤT PIPELINE DATA ENGINEERING! ==========")


if __name__ == "__main__":
    main()
