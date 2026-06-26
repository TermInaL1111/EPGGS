# EPGGS 部署、训练、实验完整文档

## 0. 论文研究问题

> 事件数据经过潜在空间对齐后，能否驱动冻结的 RGB 多视角几何 backbone (VGGT)，完成 3DGS 重建？

### 核心假设

事件数据存在于与 RGB 相同的潜在流形上 (REALM 已验证)。通过轻量的 Student ViT + Projector 将事件 token 对齐到 DINOv2 空间，可以复用冻结的 VGGT 多视角几何推理。

### 与现有工作的区别

| | EvGGS | REALM | AnySplat | EPGGS |
|:---|:---:|:---:|:---:|:---:|
| Event 输入 | ✅ | ✅ | ❌ | ✅ |
| 预训练 RGB backbone | ❌ | ✅ (单帧) | ✅ | ✅ (多视角 VGGT) |
| 输出 3DGS | ✅ | ❌ | ✅ | ✅ |
| 无位姿 | ❌ | — | ✅ | ✅ |
| 训练数据 | 79 物体 | 多数据 | 250K 场景 | 79 物体 |
| 冻结 backbone, 只训 adapter | ❌ | ✅ | ❌ | ✅ |

### 创新点

1. 首次验证事件数据可通过 REALM 式对齐驱动冻结 VGGT 完成 3DGS
2. 三层对齐策略: Token → 几何一致性 → 任务反馈
3. 仅 79 个合成物体训练，与 EvGGS 可比的重建质量

---

## 1. 架构

```
Event Voxel (B,V,8ch,336,336) → Student ViT [12层, 768维, 训练]
    → Projector [Linear+LN, 训练, 768→1024]
    → VGGTInjector [冻结, 24层交替Attention]
    → CameraToken → F_C [微调] → 位姿
    → ImageTokens → VGGT depth_head [冻结] → 深度
    → ImageTokens → Intensity UNet Decoder [训练] → 灰度
    → depth+intensity+feat → Gaussian Head [训练] → R,S,α
    → 渲染 → L_render
```

### 参数量

| 组件 | 参数 | 状态 |
|:---|:---:|:---:|
| Student ViT | ~86M | ✅ 训练 |
| Projector | ~1M | ✅ 训练 |
| VGGT-1B | ~800M | ❌ 冻结 |
| F_C 位姿头 | <1M | ✅ 微调 |
| Intensity Decoder | ~5M | ✅ 训练 |
| Gaussian Head | ~2M | ✅ 训练 |

---

## 2. 部署

```bash
# 环境
conda create -n epggs python=3.10 -y && conda activate epggs
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch_msssim opencv-python numpy huggingface_hub flash-attn gsplat einops

# VGGT 权重 (首次运行自动下载 ~3GB)
python -c "from src.model.encoder.vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')"

# 验证架构
python verify_arch.py
```

---

## 3. 训练

```bash
python train_epggs.py \
    --data_root /path/to/Ev3D-S \
    --epochs 200 --warmup_epochs 20 \
    --lr 1e-4 --batch_size 2 --num_views 3 \
    --output_dir ./output
```

**Phase 1 (Epoch 1-20):** 只训 Student + Projector, L = cos(token) + sl1(token)
**Phase 2 (Epoch 21-200):** 解冻 F_C + Intensity + Gaussian, L += L_geo + L_render

预期显存: 12-16GB, 训练时间: ~12h/200 epochs on RTX 3090

---

## 4. 实验计划 (对应 PPT outline)

| 实验 | 内容 |
|------|------|
| Baseline | EvGGS (唯一 Event 前馈 3DGS) |
| 主要对比 | EPGGS vs EvGGS on Ev3D-S |
| 消融1 | Phase1 only vs Phase1+2 |
| 消融2 | VGGT 冻结 12/18/24 层 |
| 消融3 | 单视角 vs 多视角输入 |
| 跨域 | Ev3D-S → TUM-VIE |

---

## 5. 文件清单

| 文件 | 功能 |
|------|------|
| train_epggs.py | 主训练, 三层对齐loss |
| verify_arch.py | 架构验证 |
| src/model/epggs/epggs_model.py | 主模型 |
| src/model/epggs/student_encoder.py | Student ViT |
| src/model/epggs/vggt_wrapper.py | VGGT注入器 |
| src/model/epggs/heads/intensity_head.py | 灰度UNet Decoder |
| src/model/epggs/heads/gaussian_head.py | 高斯头 (EvGGS复制) |
| src/dataset/ev3d_dataset.py | Ev3D-S数据加载 |
