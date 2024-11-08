python opensora/train/train_causalvae.py \
    --exp_name "causalvae_usc75spk_finetune_121x80x80" \
    --load_from_huggingface LanguageBind/Open-Sora-Plan-v1.1.0 \
    --cache_dir "./cache_dir" \
    --batch_size 1 \
    --precision bf16 \
    --max_steps 160000 \
    --save_steps 1000 \
    --output_dir results/causalvae_sr1 \
    --video_path /project/shrikann_35/xuanshi/DATA/SPAN/span_75speakers \
    --video_num_frames 121 \
    --resolution 80 \
    --sample_rate 1 \
    --n_nodes 1 \
    --devices 2 \
    --num_workers 8