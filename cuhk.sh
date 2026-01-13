#!/bin/bash

# 清理__pycache__缓存文件
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true

# 注意：Windows下路径分隔符建议使用 /
# 注意：Windows下 workers 建议设置较小（如 0 或 2），避免多进程死锁
# 注意：已将 root 指向 CUHK-PEDES/imgs 以避免递归搜索死锁
# 注意：已将 json_file 更新为目录中实际存在的 data_captions_ori.json
# 警告：检测到目录下缺失 caption_cloth.json 和 caption_id.json，请确保这些文件存在，否则训练会报错

python scripts/train.py \
    --root datasets \
    --dataset-configs "[{'name': 'CUHK-PEDES', 'root': 'CUHK-PEDES/imgs', 'json_file': 'CUHK-PEDES/annotations/caption_all.json', 'cloth_json': 'CUHK-PEDES/annotations/caption_cloth.json', 'id_json': 'CUHK-PEDES/annotations/caption_id.json'}]" \
    --batch-size 64 \
    --lr 0.0001 \
    --weight-decay 0.001 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 500 \
    --workers 2 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --save-freq 10 \
    --fp16 \
    --num-classes 8000 \
    --fusion-type "enhanced_mamba" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 2 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.1 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --loss-info-nce 1.0 \
    --loss-cls 1.0 \
    --loss-cloth 0.5 \
    --loss-cloth-adv 0.1 \
    --loss-cloth-match 1.0 \
    --loss-decouple 0.1 \
    --loss-gate-regularization 0.01 \
    --loss-projection-l2 0.0001 \
    --loss-uniformity 0.01 \
    --optimizer "Adam" \
    --scheduler "cosine" \
    --logs-dir log/cuhk_pedes