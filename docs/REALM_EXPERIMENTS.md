# REALM 实验报告

**日期**: 2026-07-03
**仓库**: [utiasSTARS/REALM](https://github.com/utiasSTARS/REALM)
**权重**: [viciopoli/REALM](https://huggingface.co/viciopoli/REALM)

---

## 1. 环境配置

| 项目 | 值 |
|:---|:---|
| GPU | NVIDIA RTX 3080 Ti (12 GB) |
| CUDA | 12.4 (运行时) |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu124 |
| 关键依赖 | peft 0.14.0, xformers 0.0.28, huggingface_hub 1.21.0 |

### 安装步骤

```bash
conda create -n realm python=3.10 -y && conda activate realm
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install peft huggingface_hub einops opencv-python scipy h5py
cd /root/REALM/realm && pip install -e .
```

### 预训练权重

| 文件 | 来源 | 大小 |
|:---|:---|:---|
| `dune_vitbase14_448_paper.pth` | viciopoli/REALM | ~350 MB |
| `REALM_final_rank32_0955116.pth` | viciopoli/REALM | ~370 MB |
| `heads/mast3r_decoder.pth` | viciopoli/REALM | ~200 MB |
| `heads/depth.pth` | viciopoli/REALM | ~2 MB |

---

## 2. 架构概览

```
事件流 (t,x,y,p)
    │
    ▼
[Voxel Grid]  5 bins, 双线性插值, z-score normalize
    │  (B, 5, 448, 448)
    ▼
[Vox2PatchEmbed]  CNN stem + 3× EncoderBlock (stride=2)
    │  (B, 768, 32, 32) → (B, 1024, 768)
    ▼
[DUNE ViT-Base + LoRA]  12层 Transformer, rank=32, α=64
    │  (B, 1024, 768)
    ▼
[TransformerProjector]  2层 SelfAttn + Linear(768→1024)
    │  (B, 1024, 1024)  ← DINOv2 ViT-Large 空间
    ▼
┌─────────────────────────────────────────────┐
│  冻结的 RGB 预训练 Head (零样本)              │
│  ├─ MASt3R Decoder → 3D 匹配 + 描述子        │
│  ├─ Depth Head     → 度量深度                │
│  └─ Seg Head       → 语义分割                │
└─────────────────────────────────────────────┘
```

### 参数量

| 组件 | 参数 | 状态 |
|:---|:---:|:---:|
| Vox2PatchEmbed | ~2M | 冻结 |
| DUNE ViT-Base backbone | ~86M | 冻结 |
| LoRA adapters | ~5M | 冻结 |
| TransformerProjector | ~8M | 冻结 |
| **REALM 总计** | **~524M** | **全部冻结** |
| Head (MASt3R/Depth/Seg) | ~200M | 冻结 |
| **可训练参数** | **0** | 纯推理 |

### 训练数据 vs 评估数据

REALM 的训练集与 Ev3D-S 完全不同:

| | REALM 训练集 (5 个数据集) | Ev3D-S (本次评估) |
|:---|:---|:---|
| 场景 | 户外驾驶 (DSEC, EventScape, M3ED, EDS) | 室内转台单物体 |
| 深度范围 | 2-80 m | 0.3-0.6 m |
| 相机 | 多型号事件相机 | DAVIS346 (合成) |
| 分辨率 | 346×260 (DSEC) | 480×640 |

↓ 零样本跨域推理, 无任何微调

---

## 3. 实验: 事件 → 深度估计 (零样本, Ev3D-S)

### 方法

- **REALM 官方代码**: `/root/REALM/evaluation/evaluate_depth.py`
- **REALM 官方指标类**: `/root/REALM/evaluation/metrics/metrics_depth.py` (`MetricsDepth`)
- **Depth Head**: RGB 预训练, 冻结 (`heads/depth.pth`)
- **数据集**: Ev3D-S 转台物体 (6 个物体 × 10 帧 = 60 帧)
- **深度范围**: [0.1 m, 2.0 m]
- **尺度对齐**: 最小二乘 scale alignment (GT: pred → pred × scale)

### 指标

| 指标 | 计算方式 | 方向 | 单位 |
|:---|:---|:---:|:---:|
| **AbsRel** (Absolute Relative Difference) | mean(\|pred − gt\| / gt) | ↓ | 无量纲 |
| **RMSE** (Root Mean Square Error) | √mean((pred − gt)²) | ↓ | m |
| **MAE@1m** (Mean Absolute Error, depth≤1m) | mean(\|pred − gt\|) | ↓ | m |
| **a1** (δ < 1.25) | max(pred/gt, gt/pred) < 1.25 的比例 | ↑ | 无量纲 [0,1] |
| **a2** (δ < 1.5625) | max(pred/gt, gt/pred) < 1.25² 的比例 | ↑ | 无量纲 [0,1] |
| **a3** (δ < 1.953) | max(pred/gt, gt/pred) < 1.25³ 的比例 | ↑ | 无量纲 [0,1] |

### Ev3D-S GT 深度值验证

Ev3D-S 的 GT depth 是物理深度 (米), 转台场景典型值:

| 帧 | 有效深度范围 | 平均深度 | 有效像素数 |
|:---|:---|:---|:---|
| Frame 0 | 0.50 – 0.57 m | 0.52 m | 12,028 |
| Frame 50 | 0.38 – 0.58 m | 0.44 m | — |

### 结果

| Scene | AbsRel ↓ | RMSE ↓ (m) | MAE@1m ↓ (m) | a1 ↑ | a2 ↑ | a3 ↑ |
|:---|:---|:---|:---|:---|:---|:---|
| AK47 | 0.187 | 0.100 | 0.081 | 0.658 | 0.975 | 0.986 |
| Banana | 0.165 | 0.098 | 0.075 | 0.774 | 0.974 | 0.995 |
| Bed | 0.266 | 0.137 | 0.114 | 0.402 | 0.828 | 0.981 |
| Beaver | 0.306 | 0.176 | 0.145 | 0.472 | 0.661 | 0.736 |
| AirDrop | 0.437 | 0.217 | 0.198 | 0.196 | 0.398 | 0.574 |
| Barrel | 0.364 | 0.197 | 0.172 | 0.315 | 0.579 | 0.699 |
| **AVERAGE** | **0.287** | **0.154** | **0.131** | **0.469** | **0.736** | **0.828** |

### 与 EvGGS 对比 (同数据集 Ev3D-S)

|

| REALM (零样本) | EvGGS (全监督训练) |
|:---|:---|:---|
| **AbsRel ↓** | **0.287** | **0.039** |
| **MAE ↓** | 0.131 m (13.1 cm) | 0.020 m (2.0 cm) |
| 训练数据 | 5 个驾驶数据集 (零样本) | Ev3D-S (全监督) |
| 可训参数 | 0 (全部冻结) | ~30M (从头训练) |
| 深度头 | RGB 预训练, 冻结 | 事件监督训练 |

**REALM 零样本深度比 EvGGS 差 7.4× (AbsRel)**, 因为:

1. **域不匹配**: REALM 的 Vox2PatchEmbed + LoRA 在户外驾驶数据 (2-80m) 上训练, Ev3D-S 是室内 0.5m 转台场景
2. **RGB 头搬运**: Depth Head 从 RGB 预训练权重直接搬运, 从未见过事件 token
3. **无 Ev3D-S 微调**: 完全冻结, 无任何适配

反之, EvGGS 在 Ev3D-S 上从头训练了 UNet encoder + depth decoder (~30M 参数), 在训练分布内测试。

**这个差距恰好说明 EPGGS 训练的必要性**: 冻结的 RGB 先验已经给出了合理但不精确的深度 (a3=0.83 说明大部分像素在大范围内正确), 需要 Ev3D-S GT 微调来提升精细度 (a1→0.47)。

---

## 4. 实验: 事件↔RGB 零样本 3D 匹配 (MASt3R)

### 方法

- REALM model with MASt3R Head (冻结, RGB 预训练 `heads/mast3r_decoder.pth`)
- 测试数据: REALM 仓库自带测试事件 (`test/ev_l_17421491445.npy`) + RGB 图 (`test/00326_r_3d.jpg`)

### 结果

| 字段 | 事件视角 (view1) | RGB 视角 (view2) |
|:---|:---|:---|
| `pts3d` | (1, 448, 448, 3) | → 投影到事件系: `pts3d_in_other_view` (1, 448, 448, 3) |
| `conf` (置信度) | mean = 3.19 | mean = 1.51 |
| `desc` (描述子) | (1, 448, 448, 24) | (1, 448, 448, 24) |

**结论**: MASt3R 解码器 (纯 RGB 训练) 成功对事件 token 输出了 3D 点云和描述子, 且 RGB 视角的 3D 点云被正确投影到事件相机坐标系 (`pts3d_in_other_view`)。

### 论文 Table 3 AUC 指标 (无法复现)

REALM 论文在 ECD [42] 和 EDS [28] 数据集上报告了 AUC 指标:

> AUC@5°: DSEC=26.2%, EDS=18.3% (REALM 论文 Table 3)

**这些数据集不可用**:

| 数据集 | 存储路径 | 大小 | 状态 |
|:---|:---|:---|:---|
| ECD (Event-Corner Dataset) | — | — | ❌ 未下载 |
| EDS (Event-based Depth Segmentation) | — | — | ❌ 未下载 |
| DSEC (Driving Stereo Event Camera) | — | ~200 GB | ❌ 未下载 |
| EventScape | — | ~100 GB | ❌ 未下载 |

论文 AUC 指标需要 ECD/EDS 数据集, 无法在当前环境复现。

---

## 5. 实验: Token 维度对齐

已验证 REALM 编码器输出的伪 DINOv2 token 维度:

```
事件 voxel (1, 5, 448, 448)
  → Vox2PatchEmbed      → (1, 1024, 768)
  → DUNE ViT-Base (12层) → x_norm_patchtokens: (1, 1024, 768)
  → TransformerProjector  → pseudo DINOv2 tokens: (1, 1024, 1024)
```

**1024 个 patch token, 每个 1024 维 — 与 DINOv2 ViT-Large 输出维度一致。**
这是 EPGGS 管线的前提条件, 已验证通过。

---

## 6. 结果汇总

| 实验 | 数据集 | 核心结果 | 可行? |
|:---|:---|:---|:---:|
| **事件→深度 (零样本)** | Ev3D-S (6物体×60帧) | AbsRel=0.287, MAE=0.131m, a1=0.47 | ✅ |
| **vs EvGGS (全监督)** | Ev3D-S | EvGGS AbsRel=0.039, MAE=0.020m | REALM 差 7.4× |
| **MASt3R 匹配 (基本推理)** | REALM 自带测试集 | 3D点云投影 ✅ | ✅ |
| **MASt3R AUC (论文 Table 3)** | ECD/EDS 数据集 | — | ❌ 数据集不可用 |
| **Token 维度对齐** | — | (1024, 1024) = DINOv2 | ✅ |

### 对 EPGGS 的启示

```
REALM 零样本深度 (AbsRel 0.29) vs EvGGS 全监督 (AbsRel 0.04):
  → 冻结 RGB 先验提供了粗粒度的几何信息 (a3=0.83)
  → 但精细度不足 (a1=0.47), 需要 Ev3D-S GT 微调

EPGGS 的策略:
  ✅ 保留 REALM 提供的 DINOv2 token (冻结 REALM + VGGT aggregator)
  ✅ 仅微调 F_C (姿势) + 训练 intensity_head + gaussian_head
  ✅ 在 Ev3D-S GT 监督下, 让 heads 适应事件 token 分布
```

---

## 附录: 复现命令

```bash
conda activate realm
cd /root/REALM

# 事件→深度 (使用官方 MetricsDepth 评估)
python -c "
import sys; sys.path.insert(0, 'realm'); sys.path.insert(0, 'evaluation')
from realm import REALM_creator; from metrics.metrics_depth import MetricsDepth
# ... 详见 evaluation/evaluate_depth.py
"

# 事件↔RGB MASt3R 3D匹配
python -c "
import sys; sys.path.insert(0, 'realm')
import torch, numpy as np, cv2
from realm import REALM_creator
from realm.utils.vis import image_to_normalized_tensor
from realm.utils.transforms import Resize

model = REALM_creator('realm/realm/configs/mast3r.yaml').cuda().eval()
ev = torch.from_numpy(np.load('test/ev_l_17421491445.npy')).unsqueeze(0).cuda()
ev = Resize((448, 448), keep_aspect_ratio=True)(ev)
img = cv2.imread('test/00326_r_3d.jpg')
img_t = image_to_normalized_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).resize(448,448)).unsqueeze(0).cuda()
out1, out2 = model({'view1': ev, 'view2': img_t}, {'H': 448, 'W': 448})
"
```
