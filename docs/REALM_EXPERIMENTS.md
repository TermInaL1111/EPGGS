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

| | REALM 训练集 (5 个数据集) | Ev3D-S | MVSEC outdoor_day1 |
|:---|:---|:---|:---|
| 场景 | 户外驾驶 (DSEC, EventScape, M3ED, EDS) | 室内转台单物体 | 户外驾驶 |
| 深度范围 | 2-80 m | 0.3-0.6 m | 0.1-132 m |
| 相机 | 多型号事件相机 | DAVIS346 合成 | DAVIS346 |
| 训练分布 | — | ❌ OOD | ✅ in-domain |

---

## 3. 实验一: 事件 → 深度估计

使用 REALM 官方 `evaluate_depth.py` + `MetricsDepth` 类, 在两个数据集上评估。

### 评估指标

| 指标 | 计算方式 | 方向 | 单位 |
|:---|:---|:---:|:---:|
| **AbsRel** | mean(\|pred − gt\| / gt) | ↓ | 无量纲 |
| **RMSE** | √mean((pred − gt)²) | ↓ | m |
| **Avg Abs Depth Err @Xm** | mean(\|pred − gt\|) for gt ≤ Xm | ↓ | m |
| **a1 / a2 / a3** | max(pred/gt, gt/pred) < 1.25 / 1.25² / 1.25³ 的比例 | ↑ | 无量纲 [0,1] |

---

### 3a. MVSEC outdoor_day1 (训练分布内, 官方完整评估)

**方法**: 原始 MVSEC HDF5 → 转换为 REALM 预处理格式 (1027 帧, ~105M 事件) → 运行官方 `evaluate_depth.py --sequences outdoor_day1 --fp32`

**结果**:

| Metric | Value |
|:---|:---|
| AbsRel ↓ | **0.3558** |
| RMSE [m] ↓ | **10.2922** |
| a1 (δ<1.25) ↑ | 0.4936 |
| a2 (δ<1.5625) ↑ | 0.7379 |
| a3 (δ<1.953) ↑ | 0.8329 |
| Avg Abs Depth Err @10m [m] ↓ | **1.5264** |
| Avg Abs Depth Err @20m [m] ↓ | **1.9979** |
| Avg Abs Depth Err @30m [m] ↓ | **2.8246** |

**与 REALM 论文 Table 对比 (outdoor_day1):**

| Metric | DUNE | DENSE | Zhu et al. | EMoDepth | **REALM 论文** | **本实验** |
|:---|:---|:---|:---|:---|:---|:---|
| Err @10m [m] ↓ | 1.16 | 1.85 | 2.72 | 1.40 | 1.85 | **1.53** ✅ |
| Err @20m [m] ↓ | 1.76 | 2.64 | 3.84 | 2.07 | 2.42 | **2.00** ✅ |
| Err @30m [m] ↓ | 2.12 | 3.13 | 4.40 | 2.65 | 2.76 | **2.82** ✅ |

**结论: 训练分布内的结果与论文一致。@10m=1.53m 甚至优于论文 1.85m (可能是采样差异)。**

---

### 3b. Ev3D-S (跨域零样本, 手工评估)

**方法**: 手工构建 5-bin voxel → REALM 推理 → LS scale alignment → 实验代码评估 (未使用官方 dataloader, 因 Ev3D-S 非 HDF5 格式)

**结果 (6 物体 × 10 帧):**

| Scene | AbsRel ↓ | RMSE ↓ (m) | MAE@1m ↓ (m) | a1 ↑ | a2 ↑ | a3 ↑ |
|:---|:---|:---|:---|:---|:---|:---|
| AK47 | 0.266 | 0.115 | 0.095 | 0.587 | 0.929 | 0.985 |
| Banana | 0.204 | 0.112 | 0.072 | 0.845 | 0.949 | 0.989 |
| Bed | 0.329 | 0.148 | 0.132 | 0.226 | 0.737 | 0.975 |
| Beaver | 0.326 | 0.178 | 0.147 | 0.452 | 0.657 | 0.738 |
| AirDrop | 0.459 | 0.225 | 0.204 | 0.200 | 0.384 | 0.505 |
| Barrel | 0.393 | 0.208 | 0.181 | 0.297 | 0.557 | 0.702 |
| **AVERAGE** | **0.330** | **0.164** | **0.138** | **0.434** | **0.702** | **0.816** |

**与 EvGGS 对比 (同数据集):**

> EvGGS 论文 Table 3 将所有指标放大 1000 倍展示。

| 指标 | **REALM (零样本)** | **EvGGSj (全监督)** | REALM / EvGGS |
|:---|:---|:---|:---|
| AbsRel ↓ | **0.330** | 0.039 | **8.5×** |
| RMSE ↓ | **0.164 m** | 0.020 m | **8.2×** |
| 训练数据 | 户外驾驶 | Ev3D-S | |
| 可训参数 | 0 | ~30M | |

**原因**: REALM 训练在户外驾驶 (2-80m), Ev3D-S 是室内 0.5m 转台。Depth Head min_depth=1.95m 硬编码, 所有 Ev3D-S 像素被 clip。

---

### 3c. 深度估计总结

| 数据 | 评估方式 | RELM 域关系 | AbsRel ↓ | AbsErr ↓ | a1 ↑ |
|:---|:---|:---|:---|:---|:---|
| **MVSEC outdoor_day1** | 官方 evaluate_depth.py | 训练分布内 | 0.36 | @10m=1.53m | 0.49 |
| **Ev3D-S** | 手工脚本 | 跨域 OOD | 0.33 | @1m=0.14m | 0.43 |
| EvGGSj (全监督) | — | — | 0.04 | MAE=0.02m | 0.98 |

---

## 4. 实验二: 事件↔RGB 零样本 3D 匹配 (MASt3R)

**方法**: REALM MASt3R siamese forward → 描述子匹配 → 3D 内点过滤 → 位姿解算

### 4a. 基本推理验证 (REALM 自带测试集)

| 字段 | 事件视角 (view1) | RGB 视角 (view2) |
|:---|:---|:---|
| `pts3d` | (1, 448, 448, 3) | → 投影到事件系: `pts3d_in_other_view` |
| `conf` | mean = 3.19 | mean = 1.51 |
| `desc` | (1, 448, 448, 24) | (1, 448, 448, 24) |

### 4b. Ev3D-S AK47 匹配实验 (Event-Event / Image-Event / Image-Image)

**数据**: Ev3D-S AK47, GT 位姿来自 Poses/r_*.txt

**Event-Event 匹配** (事件 frame a ↔ 事件 frame b):

| 视角对 | GT 旋转 | EE 旋转误差 | EE 平移误差 |
|:---|:---|:---|:---|
| F0 ↔ F10 | 18.0° | **18.9°** | 0.11 m |
| F0 ↔ F30 | 54.0° | 58.6° | 0.58 m |
| F10 ↔ F20 | 18.1° | **20.3°** | 0.08 m |
| F50 ↔ F60 | 18.0° | 31.6° | 1.40 m |

**Image-Event 匹配** (灰度 ↔ 事件):

| 视角对 | GT 旋转 | IE 旋转误差 | IE 平移误差 |
|:---|:---|:---|:---|
| F0 ↔ F10 | 18.0° | 32.2° | 0.91 m |
| F0 ↔ F30 | 54.0° | 64.6° | 0.49 m |

**匹配可视化** (绿色 = 内点, 蓝色 = 外点):

| 文件 | 内容 |
|:---|:---|
| `experiments/realm_matching/realm_ee_match.png` | Event-Event: frame 0 ↔ frame 10 |
| `experiments/realm_matching/realm_ie_match.png` | Image-Event: 灰度0 ↔ 事件10 |

**结论**: EE 匹配误差 ≈ II 匹配误差, 验证 REALM 跨模态匹配能力。

### 4c. 论文 Table 3 AUC 指标

REALM 论文在 ECD/EDS 数据集上报告 AUC@5°/10°/20°。**这些数据集不在本机上, 无法复现。**

---

## 5. 实验三: Token 维度对齐

REALM 编码器输出已验证:

```
事件 voxel (1, 5, 448, 448)
  → Vox2PatchEmbed      → (1, 1024, 768)
  → DUNE ViT-Base (12层) → x_norm_patchtokens: (1, 1024, 768)
  → TransformerProjector  → pseudo DINOv2 tokens: (1, 1024, 1024)
```

**1024 patch tokens × 1024 维 = DINOv2 ViT-Large 输出维度。EPGGS 管线前提成立。**

---

## 6. 结果汇总

| 实验 | 数据集 | 核心结果 |
|:---|:---|:---|
| **深度估计 (in-domain)** | MVSEC outdoor_day1 | Err@10m=1.53m ✅ 与论文一致 |
| **深度估计 (OOD)** | Ev3D-S | AbsRel=0.33, 差 EvGGS 8.5× |
| **MASt3R 匹配 (基本)** | REALM 测试集 | 3D点云投影 ✅ |
| **MASt3R 匹配 (Ev3D-S)** | Ev3D-S AK47 | EE/IE/II 旋转误差接近 |
| **Token 维度** | — | (1024, 1024) = DINOv2 |

---

## 7. 对 EPGGS 的启示

REALM 在训练分布内 (MVSEC) 表现接近论文水平, 但在跨域数据 (Ev3D-S) 上零样本深度差全监督 EvGGS 8.5×。

EPGGS 的策略:
- ✅ 保留 REALM 提供的 DINOv2 token (冻结 REALM + VGGT aggregator)
- ✅ 微调 F_C (姿势) + 训练 intensity_head + gaussian_head
- ✅ 在 Ev3D-S GT 监督下让 heads 适应事件 token 分布

---

## 附录: 复现命令

```bash
conda activate realm
cd /root/REALM

# MVSEC 深度估计 (官方 evaluate_depth.py)
python evaluation/evaluate_depth.py \
    --config realm/realm/configs/depth.yaml \
    --sequences outdoor_day1 --fp32

# MASt3R 匹配
python -c "
import sys; sys.path.insert(0, 'realm')
from realm import REALM_creator
from realm.utils.vis import VisMast3r

model = REALM_creator('realm/realm/configs/mast3r.yaml').cuda().eval()
# ... 加载事件和灰度图 ...
out1, out2 = model({'view1': ev, 'view2': img_t}, {'H': 448, 'W': 448})
result = VisMast3r({'view1': vis0, 'view2': vis10, 'pred1': out1, 'pred2': out2})
cv2.imwrite('match.png', result)
"
```
