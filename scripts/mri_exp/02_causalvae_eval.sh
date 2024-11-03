python opensora/eval/eval_common_metric.py \
    --batch_size 2 \
    --real_video_dir /project/shrikann_35/xuanshi/DATA/SPAN/span_75speakers \
    --generated_video_dir ../test_eval/release \
    --device cuda \
    --sample_fps 10 \
    --crop_size 80 \
    --resolution 80 \
    --num_frames 85 \
    --sample_rate 1 \
    --subset_size 100 \
    --metric ssim