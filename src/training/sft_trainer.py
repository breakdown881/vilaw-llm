"""
Script Huấn luyện SFT (Supervised Fine-Tuning) bằng Unsloth & QLoRA
Dự án: VILaw-LLM
File: src/training/sft_trainer.py

HƯỚNG DẪN THỰC HÀNH CHO HỌC VIÊN:
File này là khung sườn (skeleton) chuẩn module hoá.
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

# Các thư viện phục vụ Fine-tuning
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

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
# BƯỚC 1: CHUẨN BỊ VÀ ĐỊNH DẠNG DỮ LIỆU (CHATML TEMPLATE)
# ==============================================================================
def prepare_dataset(parquet_path: str, tokenizer, max_samples: int = None):
    """
    Nhiệm vụ:
    - Đọc dữ liệu từ file parquet
    - Đưa từng dòng về định dạng danh sách hội thoại chuẩn ChatML:
      [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ...},
        {"role": "assistant", "content": ...}
      ]
    - Áp dụng tokenizer.apply_chat_template để sinh chuỗi text huấn luyện
    """
    logger.info(f"Đang nạp dữ liệu huấn luyện từ: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42)

    # TODO 1: Duyệt qua từng dòng của df và đóng gói thành list các dict có key "conversations"
    # Gợi ý:
    # formatted_data = []
    # for _, row in df.iterrows():
    #     ...
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        })

    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))

    # TODO 2: Áp dụng chat template của tokenizer vào từng mẫu trong dataset
    # Gợi ý: Dùng dataset.map với hàm apply_template
    def apply_template(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["conversations"]
        ]
        return {"text": texts}
    dataset = dataset.map(apply_template, batched=True)
    
    return dataset


# ==============================================================================
# BƯỚC 2: KHỞI TẠO MODEL VÀ GẮN LORA ADAPTER
# ==============================================================================
def setup_model_and_lora(model_cfg: dict, lora_cfg: dict):
    """
    Nhiệm vụ:
    - Nạp base model Qwen2.5-7B-Instruct với 4-bit quantization qua FastLanguageModel
    - Gắn PEFT LoRA adapter (r, alpha, target_modules)
    - Cấu hình tokenizer với chat_template="chatml"
    """
    logger.info("Đang nạp Base Model với Unsloth 4-bit...")
    # TODO 3: Gọi FastLanguageModel.from_pretrained() với các tham số từ model_cfg
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model_name"],
        max_seq_length=2048,
        dtype=None,             # Tự động nhận diện kiểu dữ liệu
        load_in_4bit=True,      # Bật 4-bit quantization
    )

    logger.info("Đang cấu hình LoRA Adapter...")
    # TODO 4: Gọi FastLanguageModel.get_peft_model() để gắn LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],                   # Rank r
        lora_alpha=lora_cfg["lora_alpha"],          # Alpha = 2 * r
        lora_dropout=lora_cfg["lora_dropout"],         # Tối ưu hóa của Unsloth
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing="unsloth", # Giảm tải VRAM
        random_state=42,
    )

    # Cài đặt ChatML template
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    return model, tokenizer


# ==============================================================================
# BƯỚC 3: CẤU HÌNH VÀ TIẾN HÀNH HUẤN LUYỆN
# ==============================================================================
def train(config_path: str):
    """
    Pipeline chính để điều phối quá trình huấn luyện SFT
    """
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    export_cfg = cfg["export"]

    # 1. Setup Model & LoRA
    model, tokenizer = setup_model_and_lora(model_cfg, lora_cfg)

    # 2. Chuẩn bị Dataset
    project_root = Path(__file__).resolve().parent.parent.parent
    train_data_path = project_root / "data" / "processed" / "legal_sft_train.parquet"
    dataset = prepare_dataset(str(train_data_path), tokenizer)

    # TODO 5: Khởi tạo TrainingArguments với các siêu tham số từ train_cfg:
    # (batch_size, gradient_accumulation_steps, learning_rate, lr_scheduler_type, optim, fp16,...)
    training_args = TrainingArguments(
        output_dir=train_cfg.get("output_dir", "./vilaw-checkpoints"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        optim=train_cfg.get("optim", "adamw_8bit"),
        fp16=train_cfg.get("fp16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        report_to=train_cfg.get("report_to", "none"),
        seed=42,
    )

    # TODO 6: Khởi tạo SFTTrainer từ thư viện TRL và bắt đầu gọi trainer.train()
    logger.info("=== 5. BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN SFT ===")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        packing=train_cfg.get("packing", False),
        args=training_args,
    )
    trainer.train()

    # TODO 7: Lưu LoRA adapter sau khi huấn luyện xong bằng model.save_pretrained()
    logger.info("=== 6. LƯU LORA ADAPTER & XUẤT MODEL ===")
    adapter_dir = project_root / "vilaw-sft-lora"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"✓ Đã lưu LoRA adapter thành công tại: {adapter_dir}")
    # Merge vào base model nếu cấu hình là merged_16bit
    if export_cfg.get("save_method") == "merged_16bit":
        merged_dir = project_root / "vilaw-sft-merged-16bit"
        logger.info(f"Đang merge adapter vào base model và lưu tại: {merged_dir}...")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        logger.info("✓ Hoàn tất xuất merged model!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VILaw-LLM SFT Training")
    parser.add_argument(
        "--config",
        type=str,
        default="src/training/configs/sft_config.yaml",
        help="Đường dẫn đến file cấu hình yaml"
    )
    args = parser.parse_args()
    train(args.config)
