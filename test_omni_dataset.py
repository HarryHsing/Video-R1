from datasets import load_dataset

# 加载整个数据集（包含 train 和 validation）
dataset = load_dataset("m-a-p/OmniInstruct_v1")

# 如果你只想加载训练集
train_dataset = load_dataset("m-a-p/OmniInstruct_v1", split="train")

print(train_dataset)
print(train_dataset[0])


# Log contents
# (echo-r1) proj218:/research/d1/gds/zhxing/projects_r1/Video-R1> python test_omni_dataset.py 
# Resolving data files: 100%|██████████████████████████████████| 30/30 [00:00<00:00, 43.06it/s]Resolving data files: 100%|██████████████████████████████| 30/30 [00:00<00:00, 165782.77it/s]Loading dataset shards: 100%|███████████████████████████████| 27/27 [00:00<00:00, 791.78it/s]Resolving data files: 100%|██████████████████████████████████| 30/30 [00:00<00:00, 41.20it/s]Resolving data files: 100%|██████████████████████████████| 30/30 [00:00<00:00, 160496.33it/s]Loading dataset shards: 100%|██████████████████████████████| 27/27 [00:00<00:00, 1381.52it/s]
# Dataset({
#     features: ['answer', 'audio', 'image', 'audio_label', 'source', 'original_meta', 'question', 'options'],
#     num_rows: 84580
# })