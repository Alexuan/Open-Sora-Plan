export WANDB_KEY=""
export ENTITY="linbin"
export PROJECT="61x80x80_speechlm_bs16_lr2e-5_1img"
# accelerate launch \
    # --config_file scripts/accelerate_configs/deepspeed_zero2_config.yaml \
    python opensora/train/train_ai2v.py \
    --model LatteA2V-XL/122 \
    --text_encoder_name DeepFloyd/t5-v1_1-xxl \
    --audio_encoder_name microsoft/speechlm \
    --cache_dir "./cache_dir" \
    --dataset ai2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path "LanguageBind/Open-Sora-Plan-v1.1.0" \
    --video_data "dataset/scripts/video_mri_train.txt" \
    --image_data "dataset/scripts/img3dsnapshot_mri_train.txt" \
    --audio_data "dataset/scripts/audio_mri_train.txt" \
    --sample_rate 5 \
    --num_frames 61 \
    --max_image_size 80 \
    --gradient_checkpointing \
    --attention_mode xformers \
    --train_batch_size=16 \
    --dataloader_num_workers 8 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --mixed_precision="bf16" \
    --report_to="wandb" \
    --checkpointing_steps=200 \
    --output_dir="61x80x80_speechlm_bs16_lr2e-5_1img" \
    --allow_tf32 \
    --use_deepspeed \
    --model_max_length 300 \
    --use_image_num 0 \
    --cond_image_num 1 \
    --enable_tiling \
    --resume_from_checkpoint "latest" 
