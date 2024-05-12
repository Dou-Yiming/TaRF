export CUDA_VISIBLE_DEVICES=0
python scripts/touch_estimator_real_time.py \
    --indir /home/ymdou/datac_ymdou/TaRF/nerfstudio/outputs/touch_estimation_input_cache \
    --bg_path touch_bg/bench_outdoor_1_colmap_40_50/bg.jpg\
    --outdir outputs/touch_estimator_real_time/ \
    --ddim_steps 200 \
    --config configs/tarf.yaml \
    --diffusion_ckpt logs/tmp/last.ckpt \
    --ranking_rgb_enc_ckpt logs/tmp/last.ckpt \
    --ranking_tac_enc_ckpt logs/tmp/last.ckpt \
    --n_samples 16 \
    --scale 7.5