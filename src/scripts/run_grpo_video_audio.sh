cd src/r1-v

export DEBUG_MODE="true" # 启用 Debug 查看 RL 期间模型推理
export LOG_PATH="./debug_log_omni.txt"

# 设置 WANDB 名称
export WANDB_NAME=Qwen2.5-Omni-7B-GRPO-Audio

# 取消注释下面的行来从检查点恢复训练
# export CHECKPOINT_PATH=/research/d1/gds/zhxing/projects_r1/Video-R1/src/r1-v/log/Qwen2.5-Omni-7B-GRPO-Audio-Only/checkpoint-500

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --output_dir ./log/$WANDB_NAME \
    --model_name_or_path /research/d1/gds/zhxing/projects_r1/models/Qwen2.5-Omni-7B \
    --dataset_name /research/d1/gds/zhxing/projects_r1/datasets/AV-TAU-R1/final_train_qa_r1.json \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 8192 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal false \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 500 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 1 \
    --use_audio_in_video true \
    --enable_audio_output false
    # --resume_from_checkpoint $CHECKPOINT_PATH