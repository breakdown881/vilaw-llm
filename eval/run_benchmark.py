"""
Script Đánh Giá Mô Hình Tự Động Bằng LLM-as-Judge (Legal Evaluation Benchmark)
Dự án: VILaw-LLM
File: eval/run_benchmark.py

HƯỚNG DẪN THỰC HÀNH CHO HỌC VIÊN:
File này là khung sườn (skeleton) chuẩn LLM-as-Judge Evaluation.
Nhiệm vụ của bạn: Hoàn thiện các khối # TODO theo hướng dẫn của Mentor.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = PROJECT_ROOT / "eval" / "legal_benchmark_200.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "benchmark_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Khởi tạo LLM Judge (Tái sử dụng Groq / Gemini / OpenAI miễn phí từ file .env)
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

if PROVIDER == "gemini":
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    JUDGE_MODEL = os.getenv("MODEL_NAME", "gemini-1.5-flash")
elif PROVIDER == "groq":
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    JUDGE_MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    JUDGE_MODEL = "gpt-4o-mini"

logger.info(f"⚖️ Khởi động Giám khảo Đánh giá (LLM-as-Judge): {JUDGE_MODEL}")

# System prompt đóng vai Giám khảo Pháp lý chấm điểm khắt khe trên thang 10
JUDGE_SYSTEM_PROMPT = """Bạn là một Chuyên gia Pháp lý Tối cao và Giám khảo Đánh giá LLM.
Nhiệm vụ của bạn là đánh giá một câu trả lời của mô hình AI đối chiếu với Câu hỏi và Căn cứ Pháp lý Chuẩn (Ground Truth).

Tiêu chí chấm điểm (Thang điểm 0 - 10):
1. Citation Quality (0 - 4 điểm): Có trích dẫn chính xác số Điều, Khoản và Tên văn bản luật quy định trong Ground Truth không?
2. Legal Reasoning (0 - 3 điểm): Lập luận áp dụng điều luật vào tình huống có logic, chặt chẽ không?
3. Practical Advice (0 - 3 điểm): Có đưa ra kết luận hoặc khuyến nghị cụ thể, hữu ích không?
4. Hallucination Check (True/False): Có trích dẫn sai luật, bịa đặt điều khoản không có thật không?

Format bắt buộc: Trả về duy nhất 1 chuỗi JSON hợp lệ:
{
  "citation_score": float (0-4),
  "reasoning_score": float (0-3),
  "advice_score": float (0-3),
  "total_score": float (0-10),
  "is_hallucination": bool,
  "critique": "Nhận xét ngắn gọn lý do chấm điểm"
}
"""


# ==============================================================================
# BƯỚC 1: HÀM CHẤM ĐIỂM BẰNG LLM-AS-JUDGE
# ==============================================================================
def evaluate_single_answer(question: str, ground_truth: str, model_answer: str) -> Dict[str, Any]:
    """
    Gửi câu trả lời của model đến LLM Judge để chấm điểm
    """
    user_content = (
        f"Câu hỏi: {question}\n"
        f"Căn cứ pháp lý chuẩn (Ground Truth): {ground_truth}\n"
        f"Câu trả lời của Mô hình AI: {model_answer}\n"
    )

    try:
        # TODO 1: Gọi client.chat.completions.create với JUDGE_MODEL
        # - messages: system prompt và user_content
        # - response_format={"type": "json_object"}
        # - temperature=0.1
        # Trích xuất content từ response, json.loads và trả về dict
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)

    except Exception as e:
        logger.error(f"❌ Lỗi khi chấm điểm: {str(e)}")
        return {
            "citation_score": 0, "reasoning_score": 0, "advice_score": 0,
            "total_score": 0, "is_hallucination": False, "critique": f"Lỗi chấm điểm: {str(e)}"
        }


# ==============================================================================
# BƯỚC 2: CHẠY BỘ BENCHMARK VÀ TỔNG HỢP SỐ LIỆU
# ==============================================================================
def run_benchmark(model_name: str = "vilaw-llm-dpo"):
    logger.info(f"=== BẮT ĐẦU CHẠY BENCHMARK ĐÁNH GIÁ MÔ HÌNH: {model_name} ===")
    
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    # Danh sách câu trả lời thực tế đã qua DPO Alignment
    sample_model_answers = {
        "LEGAL_001": "Theo quy định tại Điều 429 Bộ luật Dân sự 2015, thời hiệu khởi kiện để yêu cầu Tòa án giải quyết tranh chấp hợp đồng là 03 năm kể từ ngày người có quyền yêu cầu biết hoặc phải biết quyền, lợi ích hợp pháp của mình bị xâm phạm.",
        "LEGAL_002": "Căn cứ Điều 301 Luật Thương mại 2005, mức phạt vi phạm nghĩa vụ hợp đồng hoặc tổng mức phạt đối với nhiều vi phạm do các bên thoả thuận không được vượt quá 8% giá trị phần nghĩa vụ bị vi phạm.",
        "LEGAL_003": "Theo quy định tại Khoản 2 Điều 188 Luật Doanh nghiệp 2020, doanh nghiệp tư nhân không được phát hành bất kỳ loại chứng khoán nào trên thị trường chứng khoán.",
        "LEGAL_004": "Theo Khoản 2 Điều 98 Bộ luật Lao động 2019, người lao động làm việc vào ban đêm thì được trả thêm ít nhất 30% tiền lương tính theo đơn giá tiền lương của ngày làm việc bình thường.",
        "LEGAL_005": "Theo Điều 23 Luật Đấu thầu 2023, hình thức chỉ định thầu được áp dụng trong các trường hợp cấp bách để khắc phục thiên tai, dịch bệnh hoặc gói thầu bí mật nhà nước."
    }

    results = []
    total_score_sum = 0
    hallucination_count = 0

    for item in benchmarks:
        q_id = item["id"]
        q_text = item["question"]
        gt = f"{item['ground_truth_law']} - {item['expected_answer']}"
        answer = sample_model_answers.get(q_id, "Không có câu trả lời.")

        logger.info(f"Đang chấm điểm câu: [{q_id}]...")
        
        # TODO 2: Gọi hàm evaluate_single_answer(q_text, gt, answer)
        # - Lấy total_score cộng dồn vào total_score_sum
        # - Nếu is_hallucination == True thì tăng hallucination_count lên 1
        # - Thêm chi tiết vào list results
        eval_result = evaluate_single_answer(q_text, gt, answer)
        score = eval_result.get("total_score", 0)
        is_hallu = eval_result.get("is_hallucination", False)
        
        total_score_sum += score
        if is_hallu:
            hallucination_count += 1
            
        results.append({
            "id": q_id,
            "category": item["category"],
            "question": q_text,
            "ground_truth": gt,
            "model_answer": answer,
            "evaluation": eval_result
        })

    # TODO 3: Tính toán điểm trung bình (avg_score) và tỷ lệ ảo giác (hallucination_rate)
    n = len(benchmarks)
    avg_score = round(total_score_sum / n, 2) if n > 0 else 0
    hallucination_rate = f"{round((hallucination_count / n) * 100, 1)}%" if n > 0 else "0%"

    summary = {
        "model_name": model_name,
        "total_test_cases": len(benchmarks),
        "average_score_out_of_10": avg_score,
        "hallucination_rate": hallucination_rate,
        "detailed_results": results
    }

    # Lưu kết quả
    output_file = RESULTS_DIR / "dpo_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"✓ Đã lưu kết quả Benchmark tại: {output_file}")
    
    # In bảng điểm tổng kết ra màn hình
    print("\n" + "="*60)
    print(f"📊 BẢNG KẾT QUẢ BENCHMARK: {model_name.upper()}")
    print("="*60)
    print(f"• Số lượng câu hỏi kiểm thử: {len(benchmarks)}")
    print(f"• Điểm Pháp lý Trung bình (Legal Score): {avg_score} / 10")
    print(f"• Tỷ lệ Ảo giác (Hallucination Rate):    {hallucination_rate}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_benchmark(model_name="vilaw-llm-dpo")
