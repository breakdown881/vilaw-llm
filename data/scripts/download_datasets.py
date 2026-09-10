import os
import sys
import logging
from pathlib import Path
from datasets import load_dataset

# 1. Cấu hình logging chuẩn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
# 2. Định vị thư mục data/raw an toàn
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
# Danh sách các dataset cần tải cho bài toán Legal Reasoning
DATASETS_CONFIG = [
    {
        "name": "thangvip/vietnamese-legal-qa",
        "save_name": "vietnamese_legal_qa_raw.parquet",
        "description": "Tập 9K QA hỏi đáp pháp lý Việt Nam"
    },
    {
        "name": "ntphuc149/ViBidLQA_v1",
        "save_name": "vibid_lqa_raw.parquet",
        "description": "Tập QA chuyên sâu Luật Đấu thầu Việt Nam"
    }
]
def download_and_save(dataset_info: dict):
    dataset_name = dataset_info["name"]
    save_path = RAW_DATA_DIR / dataset_info["save_name"]
    description = dataset_info["description"]
    
    logger.info(f"==> Đang tải: {dataset_name} ({description})...")
    try:
        # Tải split train
        ds = load_dataset(dataset_name, split="train")
        num_rows = len(ds)
        columns = ds.column_names
        
        logger.info(f"✓ Tải thành công {dataset_name}!")
        logger.info(f"  - Số lượng bản ghi: {num_rows:,}")
        logger.info(f"  - Các cột (columns): {columns}")
        
        # In thử 1 dòng mẫu
        sample = ds[0]
        logger.info(f"  - Bản ghi mẫu đầu tiên:\n{sample}\n")
        
        # Lưu định dạng Parquet
        ds.to_parquet(str(save_path))
        logger.info(f"✓ Đã lưu tại: {save_path}\n")
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải dataset {dataset_name}: {str(e)}")
        raise e
def main():
    logger.info("=== BẮT ĐẦU PIPELINE TẢI DỮ LIỆU THÔ ===")
    for config in DATASETS_CONFIG:
        download_and_save(config)
    logger.info("=== HOÀN TẤT TẢI DỮ LIỆU THÔ! ===")
if __name__ == "__main__":
    main()