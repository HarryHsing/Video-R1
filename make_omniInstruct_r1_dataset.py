import os
import json
import soundfile as sf
from datasets import load_dataset

from tqdm import tqdm

dataset = load_dataset("m-a-p/OmniInstruct_v1", split="train")
output_dir = "OmniInstruct_V1_R1/train"
os.makedirs(f"{output_dir}/audios", exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

final_data = []
for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
    audio_array = example["audio"]["array"]
    audio_rate = example["audio"]["sampling_rate"]
    audio_path = f"{output_dir}/audios/sample_{idx}.wav"
    sf.write(audio_path, audio_array, audio_rate)

    image_path = f"{output_dir}/images/sample_{idx}.jpg"
    example["image"].save(image_path)

    try:
        correct_idx = example["options"].index(example["answer"])
        correct_label = "ABCD"[correct_idx]
    except:
        continue

    new_item = {
        "problem_id": idx,
        "problem": example["question"],
        "data_type": "image_audio",
        "problem_type": "multiple choice",
        "qa_type": "avqa",
        "options": [
            {"label": l, "text": t}
            for l, t in zip("ABCD", example["options"])
        ],
        "solution": f"<answer>{correct_label}</answer>",
        "path": {
            "image": image_path,
            "audio": audio_path
        },
        "data_source": "OmniInstruct_v1"
    }

    final_data.append(new_item)

with open(f"{output_dir}/omni_rl_format.json", "w") as f:
    json.dump(final_data, f, indent=2)
