"""
Script Huấn luyện DPO Alignment (Direct Preference Optimization) bằng TRL & Unsloth
Dự án: VILaw-LLM
File: src/training/dpo_trainer.py

HƯỚNG DẪN THỰC HÀNH CHO HỌC VIÊN:
File này là khung sườn (skeleton) chuẩn module hoá cho DPO.
Nhiệm vụ của bạn: Hoàn thiện các khối # TODO theo hướng dẫn của Mentor.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import yaml
import torch
import pandas as pd
from datasets import Dataset

# Thư viện phục vụ DPO
from training import dpo_trainer
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig

# Kích hoạt bản vá tối ưu hóa DPO cho Unsloth
PatchDPOTrainer()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Bạn là một chuyên gia tư vấn pháp luật Việt Nam am hiểu sâu sắc các quy định pháp luật. "
    "Hãy trả lời câu hỏi dựa trên các văn bản quy phạm pháp luật hiện hành, "
    "viện dẫn chính xác số Điều, Khoản, tên luật và đưa ra lập luận logic, rõ ràng."
)


def load_config(config_path: str) -> dict:
    """Đọc file cấu hình YAML"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==============================================================================
# BƯỚC 1: ĐỊNH DẠNG DATASET DPO (PROMPT, CHOSEN, REJECTED)
# ==============================================================================
def prepare_dpo_dataset(parquet_path: str, tokenizer):
    """
    Nhiệm vụ:
    - Đọc dữ liệu từ file legal_dpo_pairs.parquet
    - Mỗi dòng phải chứa 3 trường: 'prompt', 'chosen', 'rejected'
    - Cần bọc 'prompt' trong System Prompt và ChatML template để mô hình nhận diện đúng vai trò
    """
    logger.info(f"Đang nạp dữ liệu DPO từ: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"-> Tổng số cặp DPO có sẵn: {len(df):,}")

    formatted_data = []

    # TODO 1: Duyệt qua từng dòng của df, format:
    # - prompt: format dạng hội thoại gồm system và user
    # - chosen: câu trả lời chuẩn mực
    # - rejected: câu trả lời kém chất lượng
    # Gợi ý:
    # for _, row in df.iterrows():
    #     prompt_messages = [
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": row["prompt"]}
    #     ]
    #     prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    #     formatted_data.append({
    #         "prompt": prompt_text,
    #         "chosen": row["chosen"],
    #         "rejected": row["rejected"]
    #     })
    
    for _, row in df.iterrows():
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]}
        ]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        formatted_data.append({
            "prompt": prompt_text,
            "chosen": row["chosen"],
            "rejected": row["rejected"]
        })

    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    return dataset


# ==============================================================================
# BƯỚC 2: NẠP SFT MODEL ĐÃ HUẤN LUYỆN ĐỂ TIẾP TỤC ALIGN DPO
# ==============================================================================
def load_sft_model(model_cfg: dict):
    """
    Nhiệm vụ:
    - Nạp base model kèm theo SFT LoRA adapter mà bạn đã train ở Sprint 3
    - Cấu hình để tiếp tục train lớp LoRA cho DPO
    """
    logger.info("Đang nạp SFT Model và LoRA Adapter từ Sprint 3...")
    
    # TODO 2: Gọi FastLanguageModel.from_pretrained() với đường dẫn adapter sft_adapter_path
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg.get("sft_adapter_path", "./vilaw-sft-lora"),
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dtype=None,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    return model, tokenizer


# ==============================================================================
# BƯỚC 3: KHỞI TẠO DPOTRAINER & TIẾN HÀNH ALIGNMENT
# ==============================================================================
def train_dpo(config_path: str):
    """
    Pipeline chính điều phối quá trình huấn luyện DPO
    """
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    dpo_cfg = cfg["dpo"]
    train_cfg = cfg["training"]
    export_cfg = cfg["export"]

    # 1. Nạp Model đã qua SFT
    model, tokenizer = load_sft_model(model_cfg)

    # 2. Chuẩn bị Dataset DPO
    project_root = Path(__file__).resolve().parent.parent.parent
    dpo_data_path = project_root / "data" / "processed" / "legal_dpo_pairs.parquet"
    dataset = prepare_dpo_dataset(str(dpo_data_path), tokenizer)

    # TODO 3: Cấu hình DPOConfig từ thư viện TRL
    # - beta: dpo_cfg["beta"] (0.1)
    # - learning_rate: train_cfg["learning_rate"] (5e-6)
    # - per_device_train_batch_size: train_cfg["per_device_train_batch_size"] (1)
    # - gradient_accumulation_steps: train_cfg["gradient_accumulation_steps"] (16)
    # - max_length, max_prompt_length
    dpo_config = DPOConfig(
        output_dir=train_cfg.get("output_dir", "./vilaw-dpo-checkpoints"),
        beta=float(dpo_cfg.get("beta", 0.1)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 16),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
        max_length=dpo_cfg.get("max_length", 1536),
        optim=train_cfg.get("optim", "adamw_8bit"),
        fp16=train_cfg.get("fp16", True),
        logging_steps=train_cfg.get("logging_steps", 5),
        seed=42,
    )

    # TODO 4: Khởi tạo DPOTrainer
    # Lưu ý cực quan trọng: ref_model=None (Unsloth sẽ tự động đóng băng base model làm reference)
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, # Tự động dùng SFT model đóng băng làm tham chiếu
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=dpo_config,
    )
    
    logger.info("🚀 Bắt đầu quá trình huấn luyện DPO Alignment...")
    dpo_trainer.train()

    # TODO 5: Chạy dpo_trainer.train() và lưu adapter vào thư mục dpo_adapter_dir
    dpo_adapter_path = project_root / export_cfg.get("dpo_adapter_dir", "./vilaw-dpo-lora")
    model.save_pretrained(str(dpo_adapter_path))
    tokenizer.save_pretrained(str(dpo_adapter_path))
    logger.info(f"✓ Đã lưu DPO Adapter thành công tại: {dpo_adapter_path}")
    logger.info("✓ Hoàn tất DPO Alignment!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VILaw-LLM DPO Alignment")
    parser.add_argument(
        "--config",
        type=str,
        default="src/training/configs/dpo_config.yaml",
        help="Đường dẫn đến file cấu hình yaml"
    )
    args = parser.parse_args()
    train_dpo(args.config)
