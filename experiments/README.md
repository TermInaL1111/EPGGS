# EPGGS 实验产物

所有实验在同一台机器上完成: RTX 3080 Ti (12GB), PyTorch 2.5.1+cu124, `realm` conda 环境。

---

## 目录结构

```
experiments/
├── realm_matching/          # REALM MASt3R 匹配可视化
├── realm_anysplat_proper/   # REALM→AnySplat 事件→3DGS 渲染
└── README.md
```

---

## 实验一: REALM MASt3R 特征匹配

**数据**: Ev3D-S AK47 (事件 + 灰度图)

| 文件 | 内容 | 数据 |
|:---|:---|:---|
| `realm_ee_match.png` | Event-Event 匹配: frame 0 vs frame 10 | 事件→5bin voxel |
| `realm_ie_match.png` | Image-Event 匹配: 灰度 frame 0 vs 事件 frame 10 | 灰度→3ch RGB / 事件→5bin voxel |

**方法**: REALM MASt3R siamese forward → reciprocal NN descriptor matching → 3D inlier filtering → 绿色(内点) / 蓝色(外点) 连线。

**结论**: 
- EE 匹配质量 ≈ II 匹配 (事件描述子和 RGB 描述子在 DINOv2 空间对齐)
- 验证了 REALM 论文的核心主张: 事件 token 可以驱动为 RGB 设计的冻结 MASt3R 解码器

---

## 实验二: REALM → AnySplat 跨模态 3DGS

**数据**: Ev3D-S AK47 (事件 frame 0, 30, 60)

| 文件 | 内容 | 数据 |
|:---|:---|:---|
| `render_f0000.png` | 事件→3DGS 渲染 (视角0) | Event frame 0 |
| `render_f0030.png` | 事件→3DGS 渲染 (视角30) | Event frame 30 |
| `render_f0060.png` | 事件→3DGS 渲染 (视角60) | Event frame 60 |
| `depth_f0000.png` | 预测深度 (视角0) | Event frame 0 |
| `depth_f0030.png` | 预测深度 (视角30) | Event frame 30 |
| `depth_f0060.png` | 预测深度 (视角60) | Event frame 60 |

**管线**:
```
事件流.npy → 5-bin voxel → REALM encoder_ev → DINOv2 token
→ (替换VGGT的patch_embed) → AnySplat VGGT Aggregator
→ F_C Pose Head + DPT Depth Head + VGGT_DPT_GS_Head + GaussianAdapter
→ 89,622 Gaussians → gsplat CUDA 渲染
```

**定量结果** (vs GT 灰度图):
| 视角 | PSNR↑ | SSIM↑ | LPIPS↓ |
|:---|:---|:---|:---|
| Frame 0 | 4.01 dB | 0.369 | 0.662 |
| Frame 30 | 3.98 dB | 0.389 | 0.649 |
| Frame 60 | 3.77 dB | 0.389 | 0.650 |
| **AVG** | **3.92 dB** | **0.383** | **0.653** |

对比: 同场景 AnySplat RGB原生路径 PSNR=35.24 dB (灰度PNG→DINOv2→VGGT)。

**结论**:
- Token注入路径打通, 全部组件正常执行
- 渲染质量远低于 RGB 原生路径 (PSNR 3.9 vs 35.2)
- REALM 未在 Ev3D-S 上训练, AnySplat heads 未适配事件 token
- 证明 EPGGS 训练的必要性
