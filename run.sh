#!/bin/bash
# EPGGS 一键运行脚本
# 在晨涧云/3090 GPU 上执行

set -e

# ── 1. 环境 (首次运行) ──
conda activate epggs 2>/dev/null || {
    conda create -n epggs python=3.10 -y
    conda activate epggs
    conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install pytorch_msssim opencv-python numpy matplotlib huggingface_hub peft
    pip install flash-attn --no-build-isolation
    pip install gsplat einops omegaconn
}

# ── 2. 下载预训练权重 (首次运行, ~3GB + ~500MB) ──
echo "=== Downloading REALM weights ==="
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('viciopoli/REALM', 'checkpoints/REALM_final_rank32_0955116.pth', local_dir='./checkpoints')
hf_hub_download('viciopoli/REALM', 'checkpoints/dune_vitbase14_448_paper.pth', local_dir='./checkpoints')
print('REALM downloaded')
"

echo "=== Downloading VGGT-1B ==="
python -c "
from src.model.encoder.vggt.models.vggt import VGGT
VGGT.from_pretrained('facebook/VGGT-1B')
print('VGGT-1B downloaded')
"

# ── 3. 验证架构 ──
echo "=== Verifying architecture ==="
python verify_arch.py

# ── 4. 训练 ──
echo "=== Starting training ==="
python train_epggs.py \
    --data_root ./data/Ev3D-S \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 1 \
    --num_views 3 \
    --output_dir ./output
