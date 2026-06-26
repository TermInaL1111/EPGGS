# EPGGS 部署、训练、实验完整文档

**最后更新:** 2026-06-26

---

## 0. 论文研究问题

> 事件数据经过潜在空间对齐后，能否驱动冻结的 RGB 多视角几何 backbone (VGGT)，完成 3DGS 重建？

### 核心假设

REALM [Polizzi 2026] 已验证: 事件数据存在于与 RGB 相同的潜在流形上。预训练的 REALM (viciopoli/REALM) 可以将事件体素映射为 DINOv2 风格的 token。这些伪 token 能否驱动冻结的 VGGT 多视角几何推理 + 3DGS 渲染？这是 EPGGS 要回答的问题。

### 与现有工作的区别

| | EvGGS | REALM | AnySplat | **EPGGS** |
|:---|:---:|:---:|:---:|:---:|
| Event 输入 | ✅ | ✅ | ❌ | ✅ |
| 预训练 backbone | ❌ (从头训UNet) | ✅ (DINOv2, 单帧) | ✅ (VGGT, 多视角) | ✅ (REALM+VGGT) |
| 输出 3DGS | ✅ | ❌ | ✅ | ✅ |
| 无外部位姿 | ❌ | — | ✅ | ✅ |
| 训练数据量 | 79 物体 | 5 数据集 | 250K 场景 | **79 物体** |
| 可训练参数 | ~30M | ~10M | ~886M | **~8M** |
| 冻结 backbone | ❌ | ✅ | ❌ | ✅ (REALM + VGGT 全冻结) |

---

## 1. 预训练权重来源

### 1.1 REALM (事件 → DINOv2 对齐)

| 项目 | 详情 |
|------|------|
| **HuggingFace** | `viciopoli/REALM` |
| **配置** | 仓库内 `realm/configs/` (depth/mast3r/segmentation) |
| **对齐目标** | DINOv2 ViT-Large (d=1024, patch=14) |
| **Student** | DUNE ViT-Base (d=768) + LoRA (rank=32, α=64) |
| **训练数据** | DSEC + EventScape + M3ED + EDS + EventPointMesh (见表 S1) |
| **训练** | 30 epochs, 4 GPUs, 512 batch, AdamW(lr=1e-3) |
| **Loss** | 0.1×MSE + 0.3×cos + 0.6×L1 |
| **可训练参数** | ~10M / 91M (LoRA adapters + voxel embedding) |
| **输出** | x_norm_patchtokens → projector → DINOv2 空间 token (N, 1024) |

### 1.2 VGGT-1B (多视角几何推理)

| 项目 | 详情 |
|------|------|
| **HuggingFace** | `facebook/VGGT-1B` (~3GB) |
| **架构** | 24 层 Alternating-Attention (帧内↔全局) |
| **输入** | DINOv2 token (B×V, N, 1024) |
| **输出** | 相机 token + 4 层中间 token + 图像 patch token |
| **自带 Head** | F_C (位姿), F_D (深度/DPT), point_head |

### 1.3 AnySplat 训练详情 (参考，EPGGS 不训 VGGT)

| 项目 | 详情 |
|------|------|
| **训练 VGGT?** | ✅ 训练 (仅冻结 patch_embed) |
| **GPU** | 16× NVIDIA A800 80GB |
| **迭代** | 15K steps, ~2 天 |
| **Optimizer** | AdamW, peak lr=2e-4, cosine schedule |
| **数据** | 9 个数据集 (~250K 场景): ARKitScenes, ScanNet++, DL3DV, CO3Dv2, Hypersim, Objaverse 等 |
| **Loss** | L_rgb + 0.05×Perceptual + 0.1×L_geo + 10.0×L_pose + 1.0×L_depth |
| **监督** | **VGGT 蒸馏伪标签** (无 GT 位姿/深度) |
| **精度** | bfloat16 + FlashAttention + gradient checkpointing |
| **总参数** | ~886M |

**EPGGS 不动 VGGT。** 原因: AnySplat 训练 VGGT 是因为 RGB 输入和 VGGT 同域。EPGGS 的输入是 REALM 对齐后的伪 token——如果微调 VGGT，它会去适应伪 token 的偏差，破坏原始的 RGB 多视角几何推理能力。冻结是最好的选择。

---

## 2. EPGGS 架构 (最终版)

```
Event Stream (t,x,y,p)
    │
    ├─→ [Voxel Grid]  5 bins + 3 frame = 8ch, 448×448
    │
    ├─→ [REALM encoder_ev]  DUNE ViT-Base + LoRA  [冻结, HF: viciopoli/REALM]
    │       → x_norm_patchtokens
    │
    ├─→ [REALM projector]  Linear(768→1024)  [冻结]
    │       → 伪 DINOv2 token (B*V, N, 1024)
    │
    ├─→ [VGGT Aggregator]  24层交替Attention  [冻结, HF: facebook/VGGT-1B]
    │       → camera_token + 4 intermediate layers + image_tokens
    │
    ├─→ camera_token → F_C (4层 SelfAttn)  [微调, <1M] → 位姿 p_i
    │       监督: L_pose = L2(p_i, Ev3D-S GT位姿)
    │
    ├─→ image_tokens → VGGT depth_head (DPT)  [冻结] → 深度
    │       监督: L_depth = L1(depth, Ev3D-S GT深度)
    │
    ├─→ image_tokens → Intensity UNet Decoder  [训练, ~5M] → 灰度 I
    │       监督: L_intensity = L1(I, Ev3D-S GT灰度)
    │
    └─→ [depth+I+feat] → Gaussian Head (GSRegressor)  [训练, ~2M] → R,S,α
            渲染 → L_render = MSE(渲染, GT灰度) + SSIM
```

### 参数量汇总

| 组件 | 参数 | 状态 | 来源 |
|:---|:---:|:---:|------|
| REALM encoder_ev | ~91M | ❌ 冻结 | HF: viciopoli/REALM |
| REALM projector | ~1M | ❌ 冻结 | HF: viciopoli/REALM |
| VGGT-1B | ~800M | ❌ 冻结 | HF: facebook/VGGT-1B |
| F_C 位姿头 | <1M | ✅ 微调 | VGGT 自带 |
| VGGT depth_head | ~10M | ❌ 冻结 | VGGT 自带 |
| Intensity Decoder | ~5M | ✅ 训练 | EPGGS 新增 |
| Gaussian Head | ~2M | ✅ 训练 | EPGGS 新增 |
| **总可训练** | **~8M** | | |
| **总冻结** | **~902M** | | |

---

## 3. 部署步骤

### 3.1 环境

```bash
conda create -n epggs python=3.10 -y && conda activate epggs
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch_msssim opencv-python numpy matplotlib huggingface_hub peft
pip install flash-attn --no-build-isolation   # VGGT 需要
pip install gsplat einops omegaconf            # 3DGS 渲染
```

### 3.2 下载预训练权重

```bash
# REALM (事件→DINOv2 对齐, 91M params)
python -c "
from huggingface_hub import hf_hub_download
# 下载 REALM checkpoint
hf_hub_download('viciopoli/REALM', 'checkpoints/REALM_final_rank32_0955116.pth')
hf_hub_download('viciopoli/REALM', 'checkpoints/dune_vitbase14_448_paper.pth')
print('REALM weights downloaded')
"

# VGGT-1B (多视角几何推理, ~800M params, ~3GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/VGGT-1B')
print('VGGT-1B downloaded')
"
```

### 3.3 数据准备

```
Ev3D-S 数据集结构:
/path/to/Ev3D-S/
├── train/
│   └── obj_XXX/
│       ├── events/frame*.npy       # (N,4) [t,x,y,p]
│       ├── images/frame*.png       # GT 灰度图
│       ├── depth/frame*.npy        # GT 深度图 (可选)
│       ├── poses.txt               # 位姿 per view
│       └── calib.txt               # 相机内参
└── test/
    └── ...
```

### 3.4 验证架构

```bash
python verify_arch.py
# 预期: 所有 tensor shapes 打印通过, 无 crash
```

---

## 4. 训练

### 4.1 训练策略

EPGGS 不需要 Phase 1 (Token 对齐)——REALM 已经训好了事件→DINOv2 的映射。

```
训练阶段:
    只更新 F_C(微调) + Intensity Decoder + Gaussian Head
    冻结: REALM + VGGT + depth_head

Loss: L = L_render(MSE+SSIM, GT灰度)
         + L_depth(L1, GT深度)
         + L_pose(L2, GT位姿)
         + L_intensity(L1, GT灰度)

监督: 全部来自 Ev3D-S GT (不需要蒸馏伪标签)
```

### 4.2 启动训练

```bash
python train_epggs.py \
    --data_root /path/to/Ev3D-S \
    --epochs 200 \
    --lr 1e-4 \
    --batch_size 1 \
    --num_views 3 \
    --output_dir ./output
```

### 4.3 预期

| 指标 | 值 |
|------|-----|
| 显存 | 14-18 GB (VGGT-1B ~800M + REALM ~91M) |
| 训练时间 | ~8-12h / 200 epochs (RTX 3090) |
| batch_size=1 | ✅ 可跑 (梯度累积) |

---

## 5. 实验计划

| 实验 | 内容 | 目的 |
|------|------|------|
| **Baseline** | EvGGS (唯一 Event 前馈 3DGS) | 质量上界 |
| **主要对比** | EPGGS: REALM+VGGT vs EvGGS on Ev3D-S | 验证核心假设 |
| **消融1** | 冻结 vs 微调 F_C | F_C 对伪 token 的适应性 |
| **消融2** | 冻结 vs 微调 depth_head | depth_head 是否依赖真 RGB token |
| **消融3** | 单视角 vs 多视角 (V=1,2,3,5) | 多视角几何推理的收益 |
| **跨域** | Ev3D-S 训练 → TUM-VIE 测试 | 泛化能力 |

---

## 6. 文件清单

| 文件 | 功能 |
|------|------|
| `train_epggs.py` | 主训练脚本 |
| `verify_arch.py` | 架构验证 (无 VGGT 可跑) |
| `epggs_architecture.dot` | 架构图 (Graphviz) |
| `src/model/epggs/epggs_model.py` | 主模型 |
| `src/model/epggs/vggt_wrapper.py` | VGGT 注入器 (绕过 patch_embed) |
| `src/model/epggs/heads/intensity_head.py` | 灰度 UNet Decoder |
| `src/model/epggs/heads/gaussian_head.py` | 高斯头 (EvGGS GSRegressor) |
| `src/dataset/ev3d_dataset.py` | Ev3D-S 数据加载 |
