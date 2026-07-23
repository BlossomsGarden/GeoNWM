# DA3 × RAEv2：多层 DINO 聚合与几何 Decoder 改造方案

> **目标。** 验证 RAEv2 的“多层视觉表征融合”思想，是否能提升 Depth Anything 3（DA3）在深度、ray、点云/3D Gaussian、相机与多视图几何一致性等任务上的表现。
>
> **核心立场。** 该方向值得实验，但不能把 RAEv2 的结果直接外推为“DA3 中各层 feature 裸逐元素相加，一定优于原始 DPT”。更合适的路线是：先将多层聚合视作对 DA3 现有 4 路 DPT pyramid 的补充；证实其有效后，再验证能否以“统一聚合表征 + 新几何 decoder”替代多尺度 DPT。

---

## 1. 背景与研究问题

### 1.1 RAEv2 给出的启发

RAEv2 发现，预训练视觉 encoder 的不同深度包含互补信息：

- 深层 feature 更偏全局语义和长程关系；
- 早/中层 feature 保留更多纹理、边界、布局与几何细节；
- 仅使用最后一层并非总是最优；
- 在保持 token 数与 channel 数不变的前提下，多层聚合（论文中为 MLS，multi-layer sum）能改善 reconstruction 和 generation。

设第 \(l\) 个视觉 Transformer block 的 patch feature 为：

\[
F_l\in\mathbb{R}^{B\times N\times C},
\]

RAEv2 的论文形式是：

\[
F_{\mathrm{MLS}}=\sum_{l\in\mathcal L}F_l.
\]

其重要价值不只是加法本身，而是建立了一个**融合不同深度信息、又不扩大后续模型特征预算的统一 representation**。

### 1.2 本方案要回答的问题

DA3 同样以 DINO 为 backbone，并面向对局部空间细节极为敏感的 dense geometry 问题。因此要检验：

> 是否可以将 DA3 的多层 DINO patch features 融合为统一、相机感知的几何 representation，并利用它改善 depth、ray、3D point / Gaussian 与跨视图一致性？

这不是证明“加法永远更好”，而是分解为三个可检验的子问题：

1. 多层 DINO feature 是否携带 DA3 当前 4 层 DPT pyramid 未充分利用的几何信息？
2. 在 camera-aware、multi-view attention 已存在的 DA3 backbone 中，怎样融合才不会破坏跨视图几何信息？
3. 统一的 multi-layer representation 能否在充分训练后替换或简化 DPT 多尺度 decoder？

---

## 2. DA3-Large-1.1 的相关现有结构

以下结论基于本地源码 [Depth-Anything-3-main/](../3/Depth-Anything-3-main/) 的 Large 配置与实现。

### 2.1 Backbone 与输出层

DA3 Large 的 backbone 是 24-block 的 DINOv2 ViT-L/14：

```yaml
name: vitl
out_layers: [11, 15, 19, 23]
alt_start: 8
qknorm_start: 8
rope_start: 8
cat_token: True
```

见 [da3-large.yaml](../3/Depth-Anything-3-main/src/depth_anything_3/configs/da3-large.yaml#L6-L28)。

其基础 embedding dim 为 1024。由于 `cat_token=True`，提供给 DPT 的 patch feature 是：

\[
F_l=[F_l^{\text{local}}\Vert F_l^{\text{global}}]
\in\mathbb{R}^{B\times S\times N_{\text{patch}}\times2048}.
\]

其中：

- \(B\)：batch；
- \(S\)：视图数 / 帧数；
- \(N_{\text{patch}}=(H/14)(W/14)\)；
- 前 1024 dim 是 local stream；
- 后 1024 dim 是 global stream。

DA3 输出时只对 global half 施加最终 DINO LayerNorm；对应实现在 [vision_transformer.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/dinov2/vision_transformer.py#L341-L398)。

> 因而，DA3 中的 2048-d feature 不是普通单流 DINO feature，而是 **local + global 两种语义状态的拼接**。融合设计必须尊重这一点。

### 2.2 Camera token 与 local/global attention

当 block index 达到 `alt_start=8` 时，DA3 将序列第 0 个 token 替换成 camera token：

- 有外参/内参时，使用 `CameraEnc` 编码得到的 camera condition；
- 无外部相机参数时，使用可学习的 reference/source camera token。

该替换发生在代码中的 `i == 8`，见 [vision_transformer.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/dinov2/vision_transformer.py#L314-L339)。

之后 DA3 交替使用：

- **local attention**：每个视图内独立 attention；
- **global attention**：把视图和 token 展平，进行跨视图 attention。

因此：

| 输出层 | 相对 camera token / global attention 的状态 |
|---|---|
| 3、7 | camera token 注入前，未经历 multi-view global attention |
| 11、15、19、23 | camera-aware，且是 global-attention 输出层 |

这也是直接将 `[3,7,11,15,19,23]` 六层**等权裸相加**的最大风险：它混合了 cross-view interaction 前后的 feature distribution。

### 2.3 原 DualDPT 的作用

DA3 Large 使用 `DualDPT`，输入 `dim_in=2048`。它固定接受 4 个 Transformer layers：

```python
intermediate_layer_idx = (0, 1, 2, 3)
```

见 [dualdpt.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/dualdpt.py#L72-L97)。

这四层并非原始空间分辨率不同；它们都来自 ViT 的同一 patch grid。DualDPT 会针对不同 transformer 深度将四路 feature 分别：

1. LayerNorm；
2. `1×1 Conv` 投影到 `[256,512,1024,1024]` channels；
3. 重采样为 `×4, ×2, ×1, ÷2` 四个空间尺度；
4. 用 top-down refinement / FPN 链融合；
5. 输出 depth、confidence、ray 与 ray confidence。

见 [dualdpt.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/dualdpt.py#L218-L311)。

所以 DPT 的 pyramid 不只是“多层 feature concat”；它显式承担了 dense geometry 的**多尺度恢复、边界细化和深浅层分工**。

---

## 3. 不能直接照搬 RAEv2 的原因

| 比较维度 | RAEv2 | DA3 |
|---|---|---|
| 融合目的 | 构造统一的 diffusion latent | dense geometry 的 depth/ray/3D 预测 |
| encoder | 冻结视觉 encoder | camera token、local/global attention 改造过的 DINOv2 |
| feature | 主要为同类 patch feature | local/global 拼接的 2048-d mixed representation |
| downstream decoder | RGB reconstruction decoder | 多尺度 DPT / DualDPT geometry decoder |
| concat 的主要代价 | 扩大 DiT latent，增加 diffusion 学习成本 | 一次 decoder 前投影通常可接受 |
| 主要难点 | 生成分布与计算预算 | 几何边界、局部平面、跨视图一致性、相机估计 |

**可迁移的思想：** 多个深度的视觉 feature 是互补的，值得构造统一 representation。  
**不可直接迁移的结论：** 无参数的逐元素相加必然优于原本的 DA3 多尺度 DPT。

RAEv2 自己也并非把“加法”视为普适定理：论文提出的是固定 latent footprint 下的简单有效设计，并通过完整的 decoder / diffusion training 验证。DA3 若改变 feature representation，也必须让几何 decoder 针对新 representation 重新适配。

---

## 4. 方案总览与优先级

| 方案 | 改动规模 | 是否保留原 DualDPT | 风险 | 推荐度 |
|---|---:|---:|---:|---:|
| A. 聚合残差注入现有四层 | 小 | 是 | 低 | **最高** |
| B. 四个深层 feature 内部融合后回注 | 小 | 是 | 很低 | 高 |
| C. 单聚合 feature + 新单尺度 geometry head | 中/大 | 否 | 中/高 | 用于验证核心假设 |
| D. concat + learnable projection | 中 | 可选 | 中 | 强对照 baseline |

---

## 5. 方案 A：聚合 feature 残差注入原 DualDPT（首选）

### 5.1 目的

在**不破坏 DA3 预训练 geometry decoder 路径**的前提下，检验更广层集合是否能给原有 `[11,15,19,23]` pyramid 提供补充几何信息。

### 5.2 融合定义

从 backbone 请求六层：

\[
\mathcal L=\{3,7,11,15,19,23\}.
\]

对每层独立归一化后聚合：

\[
G=
\frac{1}{\sqrt{|\mathcal L|}}
\sum_{l\in\mathcal L}
\alpha_l\operatorname{LN}_l(F_l),
\qquad
\sum_l\alpha_l=1.
\]

其中：

- \(F_l\in\mathbb R^{B\times S\times N\times2048}\)；
- `LN_l` 是每层独立 LayerNorm；
- `α_l` 可为固定均匀权重，也可设为可学习 softmax 权重；
- `1/√K` 用于让方差量级在层数变化时更稳定。

再把聚合结果作为 residual 注入原四层：

\[
\tilde F_i = F_i+\gamma_iP_i(G),
\qquad i\in\{11,15,19,23\}.
\]

其中：

- \(P_i\) 可以先取 identity，或使用每个 stage 的 `Linear(2048,2048)` / `1×1 Conv`；
- \(\gamma_i\) 是零初始化 scalar 或 channel-wise gate。

### 5.3 为什么零初始化 gate 很重要

令 \(\gamma_i=0\) 初始化，则训练刚开始时：

\[
\tilde F_i=F_i.
\]

模型精确退化为原 DA3 checkpoint 的输入分布；训练只在确认聚合 feature 有收益后才逐步打开新路径。这降低了：

- 直接改变 feature amplitude 导致预训练 head 崩溃的风险；
- 多视图 global feature 被早期 local feature 覆盖的风险；
- 大规模微调前无法定位原因的风险。

### 5.4 推荐实现位置

最简洁的实现方式是新增一个 `MultiLayerFeatureAggregator`，放在：

```text
DepthAnything3Net.forward()
    backbone(...) -> (feats, aux_feats)
    aggregator(feats) -> fused_feats
    _process_depth_head(fused_feats, H, W)
```

对应现有主路径见 [da3.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/da3.py#L125-L151)。

backbone 的 `out_layers` 需从：

```yaml
[11, 15, 19, 23]
```

改为：

```yaml
[3, 7, 11, 15, 19, 23]
```

但 aggregator 输出给 DualDPT 的仍应是四个 feature：`[11,15,19,23]` 的残差增强版本。

### 5.5 对 camera decoder 的约束

相机姿态分支目前直接读取最终输出层的 camera token：

```python
pose_enc = self.cam_dec(feats[-1][1])
```

见 [da3.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/da3.py#L211-L228)。

本方案应：

- **仅修改 patch-feature 分支** `feats[k][0]`；
- 仍然保留原 `feats[-1][1]` 作为 camera token；
- 不把 camera token 和 patch token 相加；
- 不改变 `CameraDec` 接口。

这样 depth/ray 聚合实验不会意外混入 pose decoder 改动。

---

## 6. 方案 B：只融合原有的四个 camera-aware 深层输出

### 6.1 目的

先回答一个更保守的问题：不引入 `alt_start=8` 前的纯 local feature 时，多层融合是否已有收益？

\[
\mathcal L_{\mathrm{deep}}=\{11,15,19,23\}.
\]

构造：

\[
G_{\mathrm{deep}}=
\frac12\sum_{l\in\mathcal L_{\mathrm{deep}}}
\alpha_l\operatorname{LN}_l(F_l),
\]

再以 residual 方式注入四层 DPT 输入。

### 6.2 优点

- 与预训练 DA3 DPT 最贴近；
- 每层都经过 camera token 注入和跨视图 global attention；
- 能将“多层融合是否有效”与“早期 local feature 是否有害/有益”分开。

这应当是方案 A 的先导实验或平行实验。

---

## 7. 方案 C：统一聚合表征 + 新单尺度 Geometry Decoder

### 7.1 目的

这是对核心假设最直接的验证：

> 是否能用一个多层融合后的 2048-d spatial representation，直接替换 DA3 的四路 DPT pyramid？

定义：

\[
G=
\frac1{\sqrt6}
\sum_{l\in\{3,7,11,15,19,23\}}
\operatorname{LN}_l(F_l)
\in\mathbb R^{B\times S\times N\times2048}.
\]

reshape 后：

```text
[B·S, 2048, H/14, W/14]
```

### 7.2 推荐 decoder，而非照搬 RAEv2 RGB decoder

**不建议使用 RAEv2 Stage-1 image reconstruction decoder。** 它的任务是 `DINO latent → RGB`，而 DA3 需要 dense geometry。更合适的是一个面向几何的单尺度 decoder：

```text
G [B·S, 2048, H/14, W/14]
  ↓ 1×1 Conv / norm / GELU，2048 → 512
  ↓ residual 3×3 Conv block
  ↓ bilinear upsample ×2 + residual block
  ↓ bilinear upsample ×2 + residual block
  ↓ bilinear upsample 至 H×W + residual block
  ├─ depth + depth confidence head
  └─ ray (6-d) + ray confidence head
```

若需要 GS，则应另行为 3D Gaussian 参数建立适配 head，而不是默认认为 depth head 的改动会自动迁移至 GS 分支。

### 7.3 这个方案的解释边界

若单尺度 decoder 表现不如原 DualDPT，**不能直接得出多层聚合失败**。原因可能是：

- DPT 的多尺度 refinement 对 depth boundary 恢复更重要；
- 模型容量不等价；
- 训练预算不足；
- 单尺度 head 不擅长把 1/14 网格恢复到像素级几何。

因此它用于验证“统一 representation 能否独立支持几何解码”，而不应代替方案 A/B。

---

## 8. 方案 D：学习型融合的强基线

RAEv2 的无参数 MLS 是重要 baseline，但 DA3 feature 的异质性更强。除裸相加外，至少应测试如下基线。

### D1. LayerNorm + scalar weighted sum

\[
G=\sum_l\operatorname{softmax}(a)_l\operatorname{LN}_l(F_l).
\]

优点：参数几乎为零，但不要求各层贡献相等。

### D2. Per-layer lightweight adapter + sum

\[
G=\sum_l\alpha_lP_l(\operatorname{LN}(F_l)),
\qquad P_l:\mathbb R^{2048}\rightarrow\mathbb R^{2048}.
\]

`P_l` 可为：

- bottleneck low-rank adapter；
- `1×1 Conv`；
- `Linear`；
- channel gate。

优点：允许将 pre-camera local feature 和 post-camera global feature 映射到可加的共同空间。

### D3. Concatenate + projection

\[
G=P([F_3\Vert F_7\Vert F_{11}\Vert F_{15}\Vert F_{19}\Vert F_{23}]),
\]

即 `12288 → 2048` 投影。相比 RAEv2 的 diffusion 场景，这种一次性 decoder 前投影对 DA3 未必成本过高，应该成为重要强对照。若它显著优于裸 sum，说明多层信息有用，但“无参数相加”的约束过强。

### D4. Local/global 分组融合

将 feature 显式拆成两组：

\[
G_{\mathrm{local}}=P_{\mathrm{local}}([F_3,F_7, F_{11}^{\mathrm{local}},\ldots]),
\]

\[
G_{\mathrm{global}}=P_{\mathrm{global}}([F_{11}^{\mathrm{global}},F_{15}^{\mathrm{global}},F_{19}^{\mathrm{global}},F_{23}^{\mathrm{global}}]).
\]

然后用 gate 合并，或向 DPT 不同尺度注入。该设计最尊重 DA3 的 local/global 语义差异，预期对 multi-view 几何更稳健。

---

## 9. 建议的消融矩阵

### 9.1 层集合消融

| ID | 层集合 | 融合方法 | 想检验的问题 |
|---|---|---|---|
| E0 | `[11,15,19,23]` | 原 DualDPT | 官方 baseline |
| E1 | `[11,15,19,23]` | LN + 均值 / weighted sum residual | camera-aware 深层融合是否有效 |
| E2 | `[9,11,15,19,23]` | residual | 最早 global interaction 后 feature 是否有用 |
| E3 | `[3,7,11,15,19,23]` | residual | 原始六层假设：早期局部细节是否补充几何 |
| E4 | `[3,7]` 与 `[11,15,19,23]` | local/global 分组融合 | 显式区分 local detail 与 global geometry |
| E5 | 六层 | concat + projection residual | 多层本身有效，还是裸加法过弱 |
| E6 | 六层 | 单聚合 map + 新 decoder | 统一 representation 能否替换 FPN |

### 9.2 融合操作消融

在固定层集合（建议先 `[11,15,19,23]`）下，依次比较：

1. 无融合；
2. 无归一化求和；
3. 每层 LayerNorm 后均值；
4. `1/√K` 缩放的求和；
5. learnable scalar weighted sum；
6. adapter + weighted sum；
7. concat + projection。

这能避免将“结果不好”错误归因于“多层 feature 无用”，而忽略层间尺度 / distribution mismatch。

---

## 10. 训练策略

### 阶段 I：冻结 backbone，只训练新增模块和 head

目的：低成本判断新的 unified representation 是否信息充足。

- 加载 DA3-Large-1.1 checkpoint；
- 冻结 DINO backbone；
- 方案 A/B：先训练 aggregator 的 weight、adapter、gate；必要时解冻 DualDPT；
- 方案 C/D：训练新 decoder 与 aggregator；
- 保留相机 token / `CameraDec` 原路径不变。

若方案 A 的 gate 为零初始化，开始时应复现原 checkpoint 结果，再观察 gate 是否从零产生稳定偏移。

### 阶段 II：解冻 block 8–23 + decoder

若阶段 I 有收益，再解冻：

- camera token 开始注入后的 transformer blocks；
- aggregation module；
- geometry decoder。

原因：这些 block 承担 camera-aware 与 cross-view representation；让它们适配新聚合目标，比从头解冻全模型更稳健。

### 阶段 III：全模型微调（可选）

只有在数据规模、训练预算、验证协议充分时才进行。否则极易把模型容量 / 训练时长带来的收益误判为融合设计收益。

---

## 11. 评估与判定标准

不要只看单张单目 depth 指标。DA3 的目标是 visual geometry foundation model，至少覆盖：

| 任务 | 建议指标 |
|---|---|
| 单目 / 多视图 depth | AbsRel、RMSE、δ1、深度边界误差 |
| 跨视图一致性 | 重投影深度一致性、对应点 3D consistency |
| Pose | rotation / translation AUC、intrinsics error |
| Dense reconstruction | point cloud F1、Chamfer distance、TSDF quality |
| Novel view / 3DGS | PSNR、SSIM、LPIPS |
| 稳定性 | 不同 view 数、不同 baseline、街景 / 室内 / 低纹理场景分组结果 |
| 代价 | 参数量、FLOPs、显存、inference latency |

判定时额外记录：

- 各层 `α_l` 学到的权重；
- 各 DPT stage `γ_i` 的 gate 值；
- local 与 global half 的 feature norm；
- 单视图与多视图下收益是否一致；
- `3/7` 引入后，局部边界和跨视图一致性是否呈相反趋势。

---

## 12. 关键风险与规避

| 风险 | 成因 | 缓解措施 |
|---|---|---|
| feature scale 爆炸 | 六层直接 sum 使 variance 增大 | 每层 LN；均值或 `1/√K` 缩放；gate 零初始化 |
| 混合不兼容 representation | `3/7` 在 camera/global attention 前，`11+` 在之后 | 先做 E1；使用分组融合或 per-layer adapter |
| 破坏原 checkpoint | DPT 接收的 feature distribution 突变 | residual injection；先冻结 backbone；从 `γ=0` 开始 |
| 丢失边界细节 | 单尺度 decoder 弱于 DPT FPN | 先保留 DualDPT；C 方案只作独立假设验证 |
| pose 性能下降 | 改动 camera token 语义 | pose 分支保持 `feats[-1][1]` 原样；不融合 camera token |
| 只优化 depth、损害 3DGS | depth 和 GS 共享 backbone 但 head 不同 | 单独评估 GSDPT；必要时为各 head 配置独立 adapter |
| 训练预算混淆结论 | 改结构同时大量 full fine-tune | 分阶段训练，固定训练步数和数据协议，保留 E0 baseline |

---

## 13. 推荐执行顺序

1. **复现 E0。** 固定 DA3-Large-1.1 数据、预处理、训练 / 验证配置，确认所有基线指标。
2. **E1。** 仅使用 `[11,15,19,23]` 做 LayerNorm + weighted residual aggregation；初始化为严格 baseline。
3. **E3。** 加入 `[3,7]`，仍用 residual injection；对比单目、多视图、pose-conditioned 三种设置。
4. **E4。** 若 E3 不稳定，改为 local/global 分组融合，而非直接放弃早层。
5. **E5。** concat + projection，判断瓶颈是“多层信息”还是“裸加法”。
6. **E6。** 在前述存在正信号后，训练单聚合表征的 geometry decoder，验证是否可取代 / 简化 DPT pyramid。
7. **逐步解冻。** 先新增模块与 head，再 block 8–23，最后才考虑全模型。

---

## 14. 最终建议

RAEv2 对 DA3 的价值，不在于把其 RGB decoder 原样迁入，而在于提出一条可验证路径：

\[
\text{多层 DINO feature}
\rightarrow
\text{统一、受限预算的表征}
\rightarrow
\text{为该表征重新适配的下游 decoder}.
\]

对于 DA3，推荐的第一版不是“六层相加后删除 DPT”，而是：

\[
\boxed{
\text{六层独立归一化与可学习聚合}
\rightarrow
\text{零初始化 residual 注入原四层 DualDPT 输入}
}
\]

这样可以在最大化复用 DA3 预训练 geometry capability 的同时，最干净地检验 RAEv2-inspired multi-layer representation 是否带来额外几何信息。

若该路径确实带来稳定收益，才值得进一步测试“单一聚合 map + 新 geometry decoder”是否能在不损失几何精度的情况下取代 / 简化 DA3 的 DPT pyramid。

---

## 源码依据

- DA3 Large 配置：[da3-large.yaml](../3/Depth-Anything-3-main/src/depth_anything_3/configs/da3-large.yaml)
- DA3 主网络与 depth / camera / GS 调用路径：[da3.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/da3.py)
- DINOv2 local/global attention、camera token 与 intermediate feature 输出：[vision_transformer.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/dinov2/vision_transformer.py)
- DualDPT 多尺度融合：[dualdpt.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/dualdpt.py)
- GS-DPT 多尺度解码：[gsdpt.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/gsdpt.py)
- RAEv2 多层特征说明：[RAEv2.md](RAEv2.md)
