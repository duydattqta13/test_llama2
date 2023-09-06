import transformers
import torch
import json

model = "meta-llama/Llama-2-7b-chat-hf"

# Khởi tạo mô hình Llama-2
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

input_file = "./data.jsonl" 

with open(input_file, "r", encoding="utf-8") as jsonl_file:
    lines = jsonl_file.readlines()

for line in lines:
    # Parse dòng JSON từ tệp
    json_data = json.loads(line)

    # Lấy key và danh sách mô tả từ dòng JSON
    key = list(json_data.keys())[0]  # Lấy key từ object JSON
    descriptions = json_data[key]

    # Ghép các mô tả thành một đoạn văn diễn giải
    explanation = ' '.join(descriptions)

    # Tạo một prompt bằng cách kết hợp key và đoạn văn diễn giải
    prompt = f"Tôi là một nhân viên báo cáo tài chính về tình hình doanh thu các đơn vị {key}. \
            bạn hãy giúp tôi mô tả chi tiết bằng tiếng việt {explanation} dùng các liên từ chứ \
            không làm mất nội dung mô tả"

    # Sử dụng model Llama-2 để tạo câu trả lời
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
        max_length=200,
    )

    for seq in sequences:
        print(f"Result for key '{key}': {seq['generated_text']}")