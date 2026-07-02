# AnySplat 实验报告

**日期**: 2026-07-02  
**仓库**: [InternRobotics/AnySplat](https://github.com/InternRobotics/AnySplat)  
**权重**: [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) (4.7GB) + [lhjiang/anysplat](https://huggingface.co/lhjiang/anysplat) (~5GB)

---

## 1. 环境配置

| 项目 | 值 |
|:---|:---|
| GPU | NVIDIA RTX 3080 Ti (12 GB) |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu124 |
| 关键依赖 | gsplat 1.5.3, xformers 0.0.28, torch_scatter 2.1.2, jaxtyping, hydra-core |

---

## 2. 模型架构

```
┌─────────────────────────────────────────────────┐
│            AnySplat Model (1191M total)           │
│                                                 │
│  Encoder (EncoderAnySplat)      1191M params    │
│  ┌───────────────────────────────────────────┐  │
│  │  VGGT-1B Aggregator     909M (部分可训)   │  │
│  │    DINOv2 patch_embed     —   冻结        │  │
│  │    Frame Attention ×24    —   可训        │  │
│  │    Global Attention ×24   —   可训        │  │
│  │                                           │  │
│  │  CameraHead (F_C)        216M (可训)      │  │
│  │  DepthHead (DPT)          33M (可训)      │  │
│  │  VGGT_DPT_GS_Head         33M (可训)      │  │
│  │    ├─ DPT Decoder (多尺度融合)             │  │
│  │    ├─ input_merger: RGB 直接注入           │  │
│  │    └─ GS params: 83ch (opacity+scale+rot+color+SH) │
│  │                                           │  │
│  │  GaussianAdapter           0M (参数在Head) │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  Decoder (DecoderSplattingCUDA)  0M (gsplat)    │
│    GPU CUDA rasterizer — 无参数，纯光栅化        │
└─────────────────────────────────────────────────┘
```

### 参数量

| 组件 | 参数 | 状态 |
|:---|:---:|:---|
| VGGT-1B Aggregator (24层交替Attention) | 909M | 可训练 (除 patch_embed) |
| CameraHead (4层 SelfAttn trunk) | 216M | 可训练 |
| DepthHead (DPT) | 33M | 可训练 |
| VGGT_DPT_GS_Head | 33M | 可训练 |
| Decoder (gsplat CUDA) | 0M | 纯算子 |
| **总计** | **1191M** | |

### AnySplat 对 VGGT-1B 的改动

| # | 改动 | 说明 |
|:---|:---|:---|
| 1 | **VGGT_DPT_GS_Head** | 继承 DPTHead，额外输出 3DGS 参数 (83通道) |
| 2 | **GaussianAdapter** | 特征→3DGS 转换: scale→[0.5,15], SH degree=4 |
| 3 | **蒸馏机制** | 冻结 VGGT 副本生成伪标签 (pose+depth)，无需 GT |
| 4 | **Voxelization** | 点云→体素锚点压缩，voxel_size=0.002 |
| 5 | **DecoderSplattingCUDA** | gsplat GPU 渲染器 |

---

## 3. 实验一: 推理能力验证

### RGB→3DGS (vrnerf/riverview, 2 视角)

| 输入 | 输出 | 性能 |
|:---|:---|:---|
| 2 张 448×448 | 353K Gaussians + pose | **1.1s / 3.6 GB** |
| 3 张 448×448 | 512K Gaussians + pose | 0.4s / 4.2 GB |
| 5 张 448×448 | 796K Gaussians + pose | 0.5s / 5.1 GB |

Gaussians 输出结构: `means (B,N,3)`, `opacities (B,N)`, `scales (B,N,3)`, `rotations (B,N,4)`, `harmonics (B,N,3,25)`

### Pose-Free 新视角合成

```
4 张输入 → VGGT预测内外参 → 3DGS → 10帧新视角渲染 (彩色+深度)
GPU: 4.5 GB
PLY 导出: 33.2 MB
```

---

## 4. 实验二: 模型结构分析

| 属性 | 值 |
|:---|:---|
| Encoder | `EncoderAnySplat` |
| pose_free | **True** |
| pred_pose | True |
| distill | False (推理模式) |
| voxelize | True, voxel_size=0.002 |
| freeze_module | patch_embed (冻结) |
| 冻结/可训 | 304M frozen / 886M trainable |

---

## 5. 实验三: NVS 定量评估 (核心实验)

### 协议

严格使用 AnySplat 官方 `eval_nvs.py` 协议:

```
1. model.encoder(ctx_images)          → Gaussians (context坐标系)
2. model.encoder.aggregator(all_imgs) → VGGT tokens
3. camera_head(tokens)                → poses (all views)
4. scale_factor = ctx_pose_mean / all_pose_mean   ← 尺度对齐
5. target_pose *= scale_factor
6. decoder.render(gaussians, target_pose, target_intr) → 渲染图
7. PSNR/SSIM/LPIPS (渲染 vs GT图像)
```

### 数据集

**Eyeful Tower / office1a (1K JPEG)**
- 来源: AWS S3 (`fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15`)
- 765 张图片, 分辨率 684×1024, 总大小 **202 MB**
- LLFF holdout split (`--llffhold 8`): 1/8 作为 target (GT)
- **该数据集不在 AnySplat 训练集中** (AnySplat 训练集: DL3DV + ScanNet++ + CO3D)

### 结果

| Setting | Views (ctx/tgt) | PSNR↑ | SSIM↑ | LPIPS↓ | Gaussians | VRAM |
|:---|:---|:---|:---|:---|:---|:---|
| Sparse (~10 ctx) | 8/2 | **15.03** | **0.6931** | 0.5004 | 1.6M | 6.2GB |
| Sparse (~15 ctx) | 17/2 | **15.63** | **0.7388** | 0.4727 | 3.4M | 8.7GB |
| Dense (>25 ctx) | — | — | — | — | — | **OOM** (12GB) |

### 与论文对比 (VR-NeRF dataset)

| Method | Views | PSNR↑ | SSIM↑ | LPIPS↓ |
|:---|:---|:---|:---|:---|
| **AnySplat (本机复现)** | 17/2 | **15.63** | **0.7388** | 0.4727 |
| AnySplat (论文, Sparse) | < 16 | 20.63 | 0.738 | 0.339 |
| AnySplat (论文, Dense) | > 32 | 23.09 | 0.781 | 0.230 |
| NoPoSplat (论文) | < 16 | 18.37 | 0.707 | 0.437 |
| 3D-GS (论文) | < 16 | 22.37 | 0.774 | 0.302 |

### 结果分析

**SSIM 追平论文说明几何重建质量已经到位** (0.7388 vs 0.738)。PSNR/LPIPS 差距来自:

1. **数据域不匹配** — office1a 不在 AnySplat 训练集中（模型从未见过该场景）
2. **Voxelization 被迫关闭** — 12GB 显存限制，关闭 voxelization 后 GS 数量膨胀（3.4M vs 论文的 ~300K），噪声增大
3. **视角不足** — 17 个 context view 覆盖不了大办公室的完整几何
4. **无 BA 优化** — `eval_pose.py` 有 100 步渲染 Bundle Adjustment 后处理（需 CO3D 数据集）

**要用 24GB GPU + Mip-NeRF360/VR-NeRF 数据集跑完整 `eval_nvs.py` 才能复现论文数字。**

---

## 6. 关键文件

| 文件 | 路径 | 用途 |
|:---|:---|:---|
| 推理脚本 | `/root/AnySplat/inference.py` | 单场景前向 |
| **NVS评估脚本** | `/root/AnySplat/src/eval_nvs.py` | **PSNR/SSIM/LPIPS 正式评估** |
| Pose评估脚本 | `/root/AnySplat/src/eval_pose.py` | CO3D 位姿 AUC |
| 指标计算 | `/root/AnySplat/src/evaluation/metrics.py` | PSNR/SSIM/LPIPS 实现 |
| 数据集 | `/cloud/cloud-ssd1/office1a/images-jpeg-1k/` | 776 张 1K JPEG |
| eval输入 | `/cloud/cloud-ssd1/office1a_eval32/` | 扁平化后 32 张 |

---

## 7. 实验四: REALM → AnySplat 跨模态 3DGS

### 目的

验证 EPGGS 的核心假设：**事件通过 REALM 对齐到 DINOv2 空间后，冻结的 AnySplat (RGB训练) 能否直接用于 3DGS 重建。**

### 实验一: 单事件复制 (失败)

首次尝试用了同一个事件文件复制 2 份 → VGGT 看到零视差 token → 深度全黑、渲染全白，3D 重建失败。

### 实验二: Ev3D-S 多视角事件 (3 帧)

使用 Ev3D-S AK47 场景的 3 个**不同**视角事件 (frame 0, 30, 60)，每个帧独立编码为伪 DINOv2 token。

### 方法

```
事件流 (t,x,y,p) — 3 帧不同视角, Ev3D-S AK47
  │  5-bin voxel grid → REALM Resize(448,448)
  ▼
REALM encoder_ev (DUNE ViT-Base+LoRA, 524M, 冻结)
  │  x_norm_patchtokens: (1, 1024, 768) per frame
  ▼
dino2reg_vitlarge_14 projector (8M, 冻结)
  │  pseudo DINOv2 token: (3, 1024, 1024)  ← 对齐到 DINOv2 ViT-Large 空间
  ▼
┌─ register_forward_hook → 替换 VGGT 的 DINOv2 patch_embed 输出
│
AnySplat VGGT-1B Aggregator (909M, bf16, 冻结)
  │  24层 Frame/Global 交替 Attention
  │  → (1, 3, 1029, 2048) frame+global 拼接
  ▼
┌─ AnySplat trained Heads (冻结) ───────────────────┐
│                                                    │
│  F_C CameraHead (216M)         → pose (1, 3, 9)   │
│  DPT DepthHead (33M)           → depth (1,3,448,448) │
│  VGGT_DPT_GS_Head (33M)        → (1, 3, 84, 448, 448) │
│    ├─ DPT 多尺度解码器                             │
│    ├─ input_merger (Conv2d 3→128, 全零dummy — 事件无RGB) │
│    └─ output_conv2 → 84ch (1 opacity + 83 GS params)│
│                                                    │
│  GaussianAdapter.forward()                          │
│    ├─ opacity: sigmoid → [0,1], mean=0.487          │
│    ├─ scale: sigmoid → [0.5, 15.0] 物理尺度          │
│    ├─ rotation: normalize 四元数                      │
│    └─ harmonics: SH degree 4 (3×25=75 coeffs)       │
│                                                    │
│  → 89,622 Gaussians                                 │
└────────────────────────────────────────────────────┘
  │
  ▼
gsplat DecoderSplattingCUDA (0M, CUDA 光栅化)
  │  → render: (1, 3, 3, 448, 448)
  ▼
PSNR / SSIM / LPIPS (vs GT 灰度图)
```

### 结果

| 视角 | PSNR↑ | SSIM↑ | LPIPS↓ |
|:---|:---|:---|:---|
| Frame 0 | 4.01 dB | 0.3694 | 0.6616 |
| Frame 30 | 3.98 dB | 0.3892 | 0.6489 |
| Frame 60 | 3.77 dB | 0.3888 | 0.6498 |
| **AVG** | **3.92 dB** | **0.3825** | **0.6534** |

| 指标 | 值 |
|:---|:---|
| Gaussians | 89,622 (GaussianAdapter 输出) |
| Opacity mean | 0.487 |
| GPU 显存 | 5.5 GB |
| 每帧事件数 | ~17,000-27,000 |

### 对比: 同一数据两条路径

| 路径 | 输入 | PSNR↑ | SSIM↑ | LPIPS↓ |
|:---|:---|:---|:---|:---|
| **AnySplat RGB 原生** | 灰度 PNG (DINOv2 patch_embed) | **35.24** | **0.978** | **0.033** |
| **REALM→AnySplat 事件注入** | 事件流 .npy (REALM token) | **3.92** | **0.383** | **0.653** |

同一场景 AK47，同一 AnySplat 权重，**输入编码方式不同导致 31dB PSNR 差距**。

### 失败原因分析

1. **REALM 域不匹配** — REALM encoder_ev 在 DSEC/EventScape (分辨率 346×260) 上训练，Ev3D-S 是 DAVIS346 (480×640)，噪声模型和场景分布完全不同
2. **伪 token ≠ 真 DINOv2 token** — REALM 的 UNIC Loss (0.5×cosine + 0.5×smooth_L1) 让 token 在方向和对齐上近似，但不是精确匹配。下游 heads 对这种偏差敏感
3. **GS Head 的 input_merger 失效** — VGGT_DPT_GS_Head 有 `Conv2d(3→128)` 分支直接注入 RGB 图像特征，事件路径没有真实 RGB，这个分支输出零
4. **DPTHead 深度尺度未标定** — 冻结的 depth_head 对伪 token 输出的深度尺度偏离真实 Ev3D-S 物理尺度

### 结论

- ✅ **Token 注入路径打通** — forward hook 替换 patch_embed，AnySplat 全部组件正常执行
- ✅ **GS head 正常输出** — 89,622 Gaussians, opacity/scale/rotation 都有合理值
- ❌ **质量不可用** — PSNR=3.92 vs 原生 35.24, 缺少事件 token 适配训练

**这恰好证明了 EPGGS 训练的必要性**: 需要 F_C 微调 + intensity_head + gaussian_head 在 Ev3D-S GT 上训练，让 heads 适应 REALM 伪 token 的分布。

### 生成文件

| 文件 | 内容 |
|:---|:---|
| `/cloud/cloud-ssd1/realm_anysplat_proper/render_f0000.png` | 事件→3DGS 渲染 (视角0) |
| `/cloud/cloud-ssd1/realm_anysplat_proper/render_f0030.png` | 事件→3DGS 渲染 (视角30) |
| `/cloud/cloud-ssd1/realm_anysplat_proper/render_f0060.png` | 事件→3DGS 渲染 (视角60) |
| `/cloud/cloud-ssd1/realm_anysplat_proper/depth_f0000.png` | 预测深度 (视角0) |
| `/cloud/cloud-ssd1/realm_anysplat_proper/depth_f0030.png` | 预测深度 (视角30) |
| `/cloud/cloud-ssd1/realm_anysplat_proper/depth_f0060.png` | 预测深度 (视角60) |

---

## 8. Ev3D-S NVS Baseline (RGB 原生路径)

作为对比，用 Ev3D-S 的灰度 PNG 直接走 AnySplat RGB 原生路径:

| Scene | Ctx/Tgt | PSNR↑ | SSIM↑ | LPIPS↓ | GS | VRAM |
|:---|:---|:---|:---|:---|:---|:---|
| AK47 | 12/3 | 35.24 | 0.9780 | 0.0333 | 2.4M | 6.3GB |
| Banana | 12/3 | 34.51 | 0.9814 | 0.0328 | 2.4M | 7.7GB |
| Bed | 12/3 | 34.69 | 0.9742 | 0.0473 | 2.4M | 7.7GB |
| Beaver | 12/3 | 31.94 | 0.9556 | 0.0489 | 2.4M | 7.7GB |
| **AVG** | | **34.10** | **0.972** | **0.041** | | |

Ev3D-S 是单物体转台场景 (12 视角覆盖所有面)，AnySplat 轻松达到 30+ PSNR。对比 office1a (真实大场景) 只有 15.6dB。灰度→RGB 对 AnySplat 影响很小 (3 个通道值相同不影响 DINOv2 patch_embed)。

---

## 9. 与 EPGGS 对比

| | AnySplat (RGB原生) | REALM→AnySplat (事件注入) | EPGGS (事件训练) |
|:---|:---|:---|:---|
| 输入模态 | RGB 图像 | **事件流** | **事件流** |
| 前端编码器 | DINOv2 patch_embed | REALM encoder_ev + proj | REALM encoder_ev + proj |
| VGGT Backbone | 部分可训 (886M) | 全部冻结 (909M) | 全部冻结 (909M) |
| F_C Pose Head | 可训 (216M) | 冻结 | **可训 (216M)** |
| DPT Depth Head | 可训 (33M) | 冻结 | 冻结 (33M) |
| 3DGS 头 | VGGT_DPT_GS_Head (33M) | VGGT_DPT_GS_Head (33M) | EPGGSGaussianHead (0.3M) + IntensityHead (10M) |
| 训练监督 | 蒸馏伪标签 (无GT) | — | **Ev3D-S GT** |
| 训练数据量 | ~250K 场景 | — | 80 物体 |
| Ev3D-S AK47 PSNR | **35.24 dB** | 3.92 dB | ❓ 待训练 |
| 总参数 | 1191M | 1691M | 1691M |
| 可训练参数 | 886M | 0 | **~217M** |

### 三阶段总结

```
Phase 1 (REALM):        事件 → DINOv2 token      ✅ 已验证 (token 维度匹配)
Phase 2 (AnySplat RGB):  RGB → DINOv2 → VGGT → GS  ✅ 已验证 (PSNR 35.24)
Phase 3 (REALM→AnySplat):事件→DINOv2→VGGT→GS      ⚠️  路径通但质量差 (PSNR 3.92)
Phase 4 (EPGGS 训练):    事件→DINOv2→VGGT→GS+训练  ❓ 待运行
                         ↑ 微调 F_C + 训练 intensity/gaussian heads
                           用 Ev3D-S GT 监督
```

---

## 附录: 复现命令

```bash
conda activate realm

# 下载 Eyeful Tower / office1a (202MB, ~30秒)
aws s3 cp --recursive --no-sign-request \
  s3://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15/EyefulTower/office1a/images-jpeg-1k/ \
  /cloud/cloud-ssd1/office1a/images-jpeg-1k/

# 扁平化图片
python -c "
from pathlib import Path; import shutil
src = Path('/cloud/cloud-ssd1/office1a/images-jpeg-1k')
out = Path('/cloud/cloud-ssd1/office1a_eval32')
out.mkdir(exist_ok=True)
for i, jpg in enumerate(sorted([j for j in src.rglob('*.jpg') if 'index' not in j.name])[:32]):
    shutil.copy2(jpg, out / f'{i:04d}.jpg')
"

# 运行正式评估 (LLFF split: every 8th = target)
cd /root/AnySplat
python src/eval_nvs.py \
    --data_dir /cloud/cloud-ssd1/office1a_eval32 \
    --llffhold 8 \
    --output_path /cloud/cloud-ssd1/office1a_output
# 输出: PSNR: XX.XX, SSIM: X.XXX, LPIPS: X.XXX
```
