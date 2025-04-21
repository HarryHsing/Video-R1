import os
import json
import time
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import torch

USE_AUDIO_IN_VIDEO = True

# === 配置 ===
qa_path = "/mnt/petrelfs/huxiaowei/projects/datasets/AV-TAU-R1/annotations/sft_formatted_test.json"
output_path = "./traffic_results_Qwen2.5-Omni-7B.jsonl"
model_path = "/mnt/petrelfs/huxiaowei/projects/models/Qwen2.5-Omni-7B"
base_path = "/mnt/petrelfs/huxiaowei/projects/datasets/AV-TAU-R1/"

# === 获取已处理过的 (video + type) 组合 ===
finished_keys = set()
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                key = os.path.join(base_path, item["video"]) + "--" + item["type"]
                finished_keys.add(key)
            except:
                continue
print(f"✅ Found {len(finished_keys)} finished items. Will skip them.")

# === 加载模型和处理器 ===
print("🔄 Loading model and processor...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

model.disable_talker()
print("Talker component has been disabled")
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
print("✅ Model ready.")

# === 读取 QA 数据 ===
with open(qa_path, 'r') as f:
    qa_data = json.load(f)
print(f"📚 Loaded {len(qa_data)} QA entries.")

# === 开始推理 ===
start_time = time.time()
with open(output_path, 'a') as fout:
    for idx, item in enumerate(qa_data):
        relative_path = item["video"]
        video_path = os.path.join(base_path, relative_path)

        q_type = item["type"]
        qa_pair = item["QA"][0]  # 默认一条QA

        question = qa_pair["q"]

        task_type = item["type"]

        key = video_path + "--" + task_type
        if key in finished_keys:
            continue

        system_text = (
            "You are a professional assistant trained to analyze traffic surveillance videos. "
            "Given a question and a video (including both visuals and audio) that may or may not contain an abnormal traffic event, "
            "your task is to provide a precise and thoughtful response. "
            "You may be asked to: (1) describe what is happening, (2) explain possible causes, "
            "(3) suggest appropriate responses, (4) recommend how to prevent similar incidents, or (5) locate the time of an anomaly if any exists. "
            "Base your answers strictly on the content of the video, including its audio."
        )

        # 构建对话
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_text}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # === 编码输入 ===
        try:
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = processor(text=text_prompt, audio=audios, images=images, videos=videos,
                               return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = inputs.to(model.device).to(model.dtype)

            # === 推理 ===
            text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            raw_output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # 提取 assistant 段落后内容
            if "assistant" in raw_output:
                output = raw_output.split("assistant")[-1].strip()
            else:
                output = raw_output.strip()


        except Exception as e:
            output = f"[ERROR] Failed on {video_path}: {str(e)}"
            print(output)

        # === 写入输出 ===
        cur_result = {
            "video": video_path,
            "type": q_type,
            "question": question,
            "response": output
        }
        fout.write(json.dumps(cur_result) + "\n")
        fout.flush()

        # 打印进度
        print(f"[{idx+1}/{len(qa_data)}] 📝 {video_path}")
        if (idx + 1) % 10 == 0:
            print(f"⏱️ Elapsed: {(time.time() - start_time)/60:.2f} mins")

print("🎉 Evaluation finished.")
