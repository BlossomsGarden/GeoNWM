# 从现有生成 Latent 审计到可几何寻址的全视角预测状态

> **核心问题。** 对一个六视角、轨迹条件的自动驾驶世界模型而言，最终被预测的 latent state 是否不仅能解码出看似合理的视频，还能在已知相机标定与 ego motion 下，**定位同一物理表面**？若不能，它就不是一个可几何验证、可供全视角导航使用的预测状态。
>
> **本文件的当前范围。** 本文件只推进一个可证伪的命题：审计现有 driving world model 实际生成的 native latent，判断它是否具有**几何可寻址性**（geometric addressability）。我们先测 OmniNWM 的 generated PDiT latent，再测 MagicDriveDiT 的 generated video latent；只有在这些审计显示一致的局限后，才将 DA3 / VGGT 作为几何基础模型状态的候选控制组。
>
> **本文件不声称的内容。** 当前不证明轨迹条件 state dynamics 更易学习，不证明 RGB/几何双解码，不实现新的全视角 NWM，也不报告规划或闭环驾驶收益。这些均是通过本审计门槛后才应开展的后续工作。

---

## 1. 为什么先审计“模型实际生成的状态”

全视角自动驾驶不是六条彼此独立的视频。六相机在任一时刻观察的是同一个世界；未来六视角观测应由同一场景状态、固定 sensor rig 与未来 ego trajectory 共同决定。

令六相机观测为：

\[
\mathcal I_t^{1:6}=\{I_t^1,\ldots,I_t^6\},
\]

相机内参及 ego-to-camera 外参为 \(C^{1:6}\)，ego pose 为 \(P_t\)，未来 ego 运动为：

\[
U_{t:t+K}=\{\Delta T_{t\rightarrow t+1},\ldots,\Delta T_{t\rightarrow t+K}\}.
\]

世界模型真正应预测的是某种状态：

\[
p_\theta\left(
\mathcal S_{t+1:t+K}
\mid
\mathcal S_{t-L+1:t},U_{t:t+K},C^{1:6}
\right).
\]

但“联合生成 RGB、depth、segmentation”不等价于“\(\mathcal S\) 本身是几何状态”。一个 decoder、条件分支或生成先验可以让输出视觉上合理，而其内部 final predictive latent 仍可能无法将一个 source token 检索到另一相机或另一时刻中同一物理表面的正确位置。

因此，本工作首先问：

> **现有全视角 driving world model 实际输出的 latent，能否在自身空间中、以外部标定几何为裁判，定位同一物理表面？**

这和 RAE-NWM 的核心动机不同。RAE-NWM 首先比较压缩 VAE 与 DINO token 的动作条件线性可预测性，并以长时单目导航 rollout 稳定性为主要证据。本文件先不讨论“哪种 state 更线性可预测”；它关注的是全视角 driving state 是否能完成 **cross-camera / cross-time geometric addressing**。

---

## 2. 研究范围：一个活跃命题与两个延期问题

### 2.1 Active Claim 1：native predictive state 是否几何可寻址

> **给定由 LiDAR / metric depth、相机标定和 ego pose 构造的真实 source-target 几何对应，候选 native predictive state 能否在完整 target state map 中准确定位同一物理表面，而非仅响应于语义或外观相似的位置？**

这个命题的审计对象首先是**已经生成出来的状态**，而不是只从 ground-truth 图像抽取 encoder feature。

### 2.2 Deferred Claim 2：哪种通过审计的 state 更易进行轨迹条件预测

若某一 state 在 Claim 1 中已无法可靠定位几何对应，直接比较其轨迹条件 dynamics 是否“更易学习”意义有限。因此本轮不训练 linear probe、mini DiT 或完整 dynamics model。

只有当 DA3 / VGGT 等候选状态在相同协议下稳定通过 Claim 1，才讨论：

\[
\hat{\mathcal S}_{t+k}^{1:6}=
f_\theta(\mathcal S_{t-L+1:t}^{1:6},\Delta T_{t\rightarrow t+k},C^{1:6}).
\]

届时应单独验证 trajectory shuffle、reverse trajectory、rig shuffle、训练效率、rollout error growth 与计算代价；这些不是当前文档的结论。

### 2.3 Not claimed：当前不证明的事项

以下事项均不属于当前 audit：

- 从 predicted state 解码高质量 RGB、depth、ray、occupancy 或 3D Gaussian；
- 端到端训练新的全视角 flow-matching / diffusion world model；
- 以 planner、MPC/CEM、counterfactual ranking 或 closed-loop 驾驶证明效用；
- 宣称任意 VAE 或 video VAE 天生无法表达几何；
- 宣称 DA3 / VGGT 必然优于所有 driving latent。

---

## 3. 审计优先的研究动机

### 3.1 Stage A：先审计 OmniNWM 的 generated PDiT state

OmniNWM 已经联合生成 RGB、depth 与 segmentation，并使用 camera ray、跨视角通信和 trajectory-related condition。它是最合适的第一审计对象：如果这样一个多模态、全视角 driving world model 的**最终生成 latent**仍不可几何寻址，那么“输出有多模态几何结果”本身不足以说明其预测空间已形成可供后续全视角 NWM 使用的几何状态。

需要审计的 state 不是笼统的“OmniNWM VAE latent”，而是：

\[
Z^{\mathrm{PDiT}}_{\mathrm{gen}}
=
\text{final sampled PDiT/MMDiT state before VAE decode}.
\]

在官方 33 帧六视角配置下：

```text
6 cameras, 448×800 RGB, 33 decoded frames
causal temporal reduction = 4

combined final PDiT state: [6, 48, 9, 56, 100]
RGB / depth / segmentation diagnostic chunks: [6, 16, 9, 56, 100] each
decoded RGB / depth / segmentation: [6, 3, 33, 448, 800] each
```

这里 combined state 是主要审计对象；RGB/depth/seg chunks 只是对同一 state 的诊断视图，不是三个统计独立模型。

官方推理配置见 [infer.py](OmniNWM-master/configs/inference/infer.py)：它指定 6 个 camera order、`num_frames=33`、448×800、`temporal_reduction=4`、PDiT depth 与 cross-view layers。推理入口在 [tools/inference.py](OmniNWM-master/tools/inference.py)，最终 PDiT latent 的未来保存点应位于 [sampling.py](OmniNWM-master/omninwm/utils/sampling.py) 中 `x = unpack(...)` 之后、`x.chunk(3, dim=1)` 和 VAE decode 之前。

### 3.2 Stage-A 失败时能说明什么，不能说明什么

若 generated PDiT state 在本文件的 geometry-localization audit 中失败，受限结论只能是：

> **在所评估的 OmniNWM checkpoint、采样设定和数据协议下，联合 RGB/depth/seg prediction 以及几何 / ray / trajectory condition，并不自动保证 final predictive latent 是几何可寻址的。**

不能由此推出：

- “VAE 天生不能表达 geometry”；
- “OmniNWM 完全没有几何信息”；
- “生成的 RGB/depth/seg 一定不一致”；
- “任何联合多模态视频生成都是 ill-posed”。

失败的竞争解释可能包括：时空压缩、重建/生成目标、几何 condition 只作为 control 而不是 state constraint、PDiT sampling error、transition dynamics、channel layout、特征归一化或 retrieval protocol。本审计的价值在于把这些可能性明确暴露出来，而不是将训练慢或输出效果间接解释为“latent 必然缺几何”。

为了避免错误归因，Stage A 必须包含两个诊断：

1. **Ground-truth encoded vs. generated state。** 对同一 VAE / modality interface，分别评估真实观测编码得到的 latent 与 PDiT generated latent。若前者也不可定位，才支持 encoding space 本身的局限；若前者可定位而后者失败，则更可能是 prediction / sampling 破坏了几何可寻址性。
2. **Combined vs. modality chunks。** 比较 combined PDiT 与 RGB/depth/seg chunks，判断地址信息是否只局部存在于某一诊断子空间。但它们属于同一模型状态，不能被包装成独立 baseline，也不能据此单独宣称某个模态“解决”了问题。

decoded depth、RGB video 与现有 consistency metrics 仅作辅助诊断。若 decoded output 看似一致、但 native PDiT state 不可定位，合理表述是：

> 输出层、条件路径或生成先验可以产生有用的几何结果，但该 final predictive state 本身未呈现可被直接检索的几何寻址结构。

### 3.3 Stage B：MagicDriveDiT 的 geometry-conditioned video latent 是独立控制

第二阶段使用 MagicDriveDiT 的**官方 native generated video latent**，以及其公开的几何控制条件（例如 camera、ego trajectory、road/BEV map、3D boxes，具体以可复现的 release interface 为准）。目的不是将它与 OmniNWM 伪装成完全相同的模型，而是检查：

> 当 driving video generation 改为另一种 VAE / DiT / 几何控制接口时，native generated latent 是否仍表现出相同的几何 addressability 局限？

实施前必须确认：

- 官方 checkpoint 与许可证；
- 使用的精确 VAE、latent scaling、空间 / 时间 layout；
- 几何 condition 的定义；
- 与外部 calibration、depth/LiDAR GT 对齐的样本协议。

若任一项不可复现，应明确标记 `unavailable`，不得用通用 SD-VAE、Hunyuan VAE 或自行替换的 latent 代替。

若 OmniNWM 与 MagicDriveDiT 都显示相同模式，正确表述是：

> 所测的 generated video-latent interfaces 共同缺乏本协议定义的 geometry addressability；这提出了一个关于 video-latent prediction spaces 的研究假设，而不是关于 Video VAE 理论不可能性的定理。

### 3.4 Stage C：为什么此后才看 DA3 / VGGT，而不是直接用 DINO

DINOv2 提供强大的 dense visual prior，但默认逐视图编码；它没有在当前六相机 rig、相机标定、跨视角通信与深度/ray/point/camera supervision 下被训练为同一物理世界的联合状态。

DA3 / VGGT 都建立在 DINO prior 之上，但引入了额外的多视角几何过程：

```text
DINOv2
→ 六视角 joint processing
→ camera / ray / pose condition
→ local-global or frame-global attention
→ depth / point / ray / camera geometry supervision
→ geometry-oriented dense feature state
```

因此，Stage C 不是假定“更重的 DA3 一定更好”，而是检验下列选择：

> 跨视角几何绑定应该由下游 driving world model 在 generated video latent 中重新在线学习，还是应先由 geometry foundation model amortize 到 encoder state？

DA3-DPT 的候选 state 严格定义为四个 DA3 intermediate layers 的 DPT stage projection、在 resize 前拼接：

\[
F_l\in\mathbb R^{B\times6\times N\times2048}
\xrightarrow{\mathrm{LN}+1\times1\ \mathrm{Conv}}
G_l\in\mathbb R^{B\times6\times C_l\times H/14\times W/14},
\]

\[
(C_{11},C_{15},C_{19},C_{23})=(256,512,1024,1024),
\]

\[
G_{\mathrm{DA3\text{-}DPT}}
=[G_{11}\Vert G_{15}\Vert G_{19}\Vert G_{23}]
\in\mathbb R^{B\times6\times2816\times H/14\times W/14}.
\]

它不是纯 frozen DINO feature，必须如实称为 `DA3-DPT state`。同时保留 `S-DA3-Raw`（四个原始 2048-d DA3 outputs 的 concat）来区分 multi-view geometry backbone 与已训练 DPT projection 的作用。DA3 的 stage projection 顺序为 token LayerNorm → stage-specific `1×1` projection → scale resize，见 [dualdpt.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/dualdpt.py#L75-L97) 和 [dualdpt.py](../3/Depth-Anything-3-main/src/depth_anything_3/model/dualdpt.py#L215-L228)。

VGGT 是同一协议下的另一类 geometry-foundation-state candidate；DINOv2 final patch feature 则是更轻的 visual-prior control。二者均不是预设赢家。

### 3.5 GLD 与 DVGT-2 的作用边界

GLD 等工作提供外部动机：DA3 / VGGT feature spaces 可以支持生成，并在冻结 decoder 下读出几何结果。这说明几何基础表征空间**值得**作为 Stage-C control 审计；它不构成自动驾驶、OmniNWM PDiT 或本文件 Claim 1 的直接证据。

DVGT-2 或其他 action-conditioned geometric-state 工作同样仅说明：若某种 geometry foundation-model state 通过本审计，未来研究其 action / trajectory-conditioned dynamics 是合理的。它不证明本项目的 Deferred Claim 2，也不证明全视角 driving NWM 已经可行。

### 3.6 条件式研究结论

只有满足下列条件，才有理由进入“直接在预训练 geometry foundation-model state 中生成”的全视角 NWM 设计：

```text
OmniNWM generated PDiT audit
        ↓
MagicDriveDiT generated-latent audit
        ↓
DA3 / VGGT candidate controls
        ↓
DA3 / VGGT 在完全相同的外部几何审计中稳定更可寻址
```

即使达到这一门槛，后续仍必须独立证明：state dynamics、计算/存储成本、紧凑 tokenizer、decode、trajectory controllability 与驾驶效用。当前不预先宣称这些结果。

---

## 4. 候选状态与原生空间比较规则

### 4.1 State family 与角色

| ID | native state | 当前角色 |
|---|---|---|
| `S-OmniNWM-PDiT-generated` | final sampled combined PDiT state before VAE decode | Stage-A 主要审计对象 |
| `S-OmniNWM-PDiT-encoded` | 与该接口对应的 ground-truth encoded latent | Stage-A 诊断：区分 encoder space 与 generation failure |
| `S-OmniNWM-PDiT-rgb/depth/seg` | 同一 generated PDiT 的三个 16-channel chunks | Stage-A 诊断视图，非独立 baseline |
| `S-MagicDriveDiT-generated` | 官方 interface 的 final generated native video latent | Stage-B 独立 driving-latent control |
| `S-DINOv2` | 每相机 DINOv2 final patch feature | 更轻的单视图 visual-prior control |
| `S-DA3-Raw` | 4 个 DA3 2048-d intermediate outputs concat | Stage-C geometry backbone control |
| `S-DA3-DPT` | DA3 DPT stage projections `[256,512,1024,1024]` concat | Stage-C 主 geometry-state candidate |
| `S-VGGT` | 官方可复现的 multiview geometric feature state | Stage-C 独立 geometry-state candidate |

DiST-4D 不纳入当前主线，除非先确认其权重、latent interface、数据对齐和评测 protocol 均可复现。

### 4.2 不做 PCA，也不强制统一通道数

不同 state 的 channel 数、spatial stride、temporal compression 与 token 数正是待研究的设计差异。对于 Claim-1 retrieval：

- 每种 state 保留原生 channel、原生 spatial grid 与原生 temporal grid；
- 仅在 state 自身的 token channel 维做 \(\ell_2\) normalization；
- source token 只与**同一种 state**的完整 target map 计算 cosine；
- 跨模型比较的是外部几何 GT 下的 localization outcome，而不是 cosine 数值；
- 不做 PCA，不把 retrieval 重采样到 DA3 grid，也不把 DA3 降到 VAE 的 channel 数。

对 state \(m\)，source token \(z_{s,m}(p_s)\) 在 target map 的相似度为：

\[
A_m(p)=
\frac{
\langle z_{s,m}(p_s),z_{t,m}(p)\rangle
}{
\|z_{s,m}(p_s)\|_2\,\|z_{t,m}(p)\|_2},
\qquad p\in\Omega_{t,m},
\]

\[
\hat p_{t,m}=\arg\max_{p\in\Omega_{t,m}}A_m(p).
\]

每种 state 必须单独报告：channel、stride、time bins、tokens/frame、scalars/frame、storage bytes、encoder/state extraction FLOPs 和 latency。它们解释质量–资源 trade-off，但不参与 similarity 的人为压缩。

---

## 5. Active Claim 1：geometry-grounded native-state localization audit

> **命题。** 对于外部标定所确定的同一物理表面，候选 native predictive state 是否能在另一相机或另一时刻的完整 native state map 中定位其真实对应位置？

### 5.1 三个递进的 source-target setting

所有 query 都基于外部 LiDAR 或经过验证的 metric depth、相机内外参和 ego pose 自动构造；不允许人工随意点选，也不允许由 feature nearest-neighbor 自举 GT。

| Setting | source → target | 状态 |
|---|---|---|
| A. 同 bin、相邻相机 overlap | \((v,\tau)\rightarrow(v',\tau)\) | 首要实验；测 rig 内同一时刻共享可见表面 |
| B. 同相机、跨 native time bins | \((v,\tau)\rightarrow(v,\tau+\Delta\tau)\) | 首要实验；仅 static、visible、unoccluded 表面 |
| C. 跨相机且跨 time | \((v,\tau)\rightarrow(v',\tau+\Delta\tau)\) | 仅在 A/B 的标定与 temporal mapping 均验证后追加 |

对任一 source image coordinate \(p_s\)：

\[
X=\Pi^{-1}(p_s,D_s(p_s)),
\qquad
p_t^\star=\pi(K_tT_{t\leftarrow s}X).
\]

仅保留以下 query：source 和 target 均在图像范围内、点位于相机前方、GT 深度有效、可见性 / z-buffer / depth consistency 合格、不是不受控动态表面，并且可映射到 source/target state 的 native grid。

### 5.2 PDiT 33 decoded frames 对 9 latent bins 的时间轴风险

OmniNWM 在 33 decoded frames 下具有 \(T_{latent}=9\) 个 causal latent bins。不能把像素视频的任意第 \(t\) 帧直接当作 PDiT 的第 \(\tau\) 个 bin。

未来 evaluator 必须：

1. 记录 causal VAE 的 temporal receptive field 与 bin-to-decoded-frame mapping；
2. 明确每个 latent bin 对应的 decoded-frame range 或预先定义的 representative frame；
3. 在 synthetic indices 上单元测试这个 mapping；
4. 同时按 `native-bin interval` 与 `decoded-frame interval` 报告跨时间结果；
5. 单独报告 temporal boundary bins 的敏感性。

若无法验证该 mapping，则不对跨时间 localization 作实质结论；同 bin 跨相机 setting 仍可独立进行。

### 5.3 ReNoV Figure 3 风格的 Similarity Atlas

每个预注册 query 面板展示：

```text
source decoded RGB + 蓝点：source token 对应的 image coordinate
target decoded / reference RGB + 绿圈：外部 GT 投影位置
灰色 mask：无效、遮挡或图像外 target 区域

combined native-state heatmap + 红叉 argmax
RGB / depth / segmentation chunk heatmap + 红叉（仅 OmniNWM 诊断）
可选：MagicDriveDiT / DINOv2 / DA3-DPT / VGGT 原生 heatmap

附注：view pair、native time bins、decoded-frame mapping、top-1 error、GT rank、validity reason
```

图像 heatmap 可双线性上采样到输入分辨率以便阅读，但 cosine、argmax、rank 必须始终在原生 token grid 上计算。

样本以固定随机种子在预定义类别中抽取：重复车道线、相似车辆、车灯/窗户、护栏、建筑立面、远距离小目标、转弯或变道。必须同时显示成功、困难成功和失败样例，禁止事后挑选最佳图。

### 5.4 轻量定量指标

主指标统一回到输入图像坐标。将 argmax token center 映射回原图后，与外部 geometry GT 的 \(p_t^\star\) 计算误差，因此更细 grid 不会因阈值定义而获得隐藏优势。

| 指标 | 含义 |
|---|---|
| image-space PCK@8 / 16 / 32 px、PCK-AUC | 是否落在共同图像坐标下的正确几何邻域 |
| median / mean top-1 localization error | 最强响应距离 GT 的像素误差 |
| GT-token rank percentile | GT 映射到各 native grid 后，在完整 target retrieval 中的相对排名 |
| MRR | GT token 在全图排序中的平均倒数排名 |
| Recall@top-1% / top-5% | 避免不同 native token 数下固定 top-K 的不公平 |
| valid-query count 与 filter reasons | 防止不同相机对或时间差因有效点数量不同造成误读 |
| state profile | 通道、grid、time bins、token/bytes、提取 latency/FLOPs；作为成本背景 |

每项主指标按相机对、time-bin separation、decoded-frame separation、近/中/远距离与预定义难例类型报告，并对 scene-level independent samples 给 bootstrap confidence interval。

### 5.5 Decoded-output consistency 只能作辅助诊断

OmniNWM 已有 [consistency_suite.py](OmniNWM-master/eval/consistency_suite.py)，可针对保存的视频/深度输出计算 feature、depth reprojection 与 temporal consistency proxy。它们可以回答“decoded output 是否存在明显不一致”，但不能代替 native PDiT geometry-localization evaluator。

两类结果应分开解释：

| native-state retrieval | decoded output diagnostic | 合理解释 |
|---|---|---|
| 好 | 好 | state 与输出均显示几何一致证据 |
| 差 | 好 | decoder / condition / prior 可产生输出一致性，但 final state 不具直接几何 addressability |
| 好 | 差 | state 可能保留对应性，但 decoder 或生成输出损失几何质量 |
| 差 | 差 | 需继续检查 state、sampling、数据或标定链路 |

---

## 6. 执行门槛与停止条件

### Gate 0：capture validity（未来实现前置检查）

本轮不实现 latent capture 或 evaluator。将来实现时，必须先验证：

- final PDiT state 在 `unpack` 后、VAE decode 前保存，而不是保存 initial noise、packed token 或 decoded output；
- combined state 与三个 16-channel chunks 可无损重构；
- 每个样本的 33 decoded frames、6-camera view order、latent shape 和 temporal mapping 均记录；
- 视频、latent、camera metadata、trajectory、source path、checkpoint/config/seed 使用 sample-unique artifact directory，不发生覆盖；
- distributed inference 只由 saving process 写入；
- raw metadata 坐标系与 calibration convention 可被复核。

现有 [tools/inference.py](OmniNWM-master/tools/inference.py) 与 [inference.py](OmniNWM-master/omninwm/utils/inference.py) 已保存可视化视频和部分 raw depth/camera data；后续扩展必须修复其多样本使用固定 `raw/` 文件名可能覆盖的问题。

### Gate 1：OmniNWM PDiT audit

使用官方 33-frame / 6-camera configuration 的独立 audit config，固定小规模 scene-diverse pilot 后再扩大样本。远程环境路径和 conda environment 仅参照 [REMOTE_SERVER.md](OmniNWM-master/REMOTE_SERVER.md)；不得在论文、日志、配置或本文档复制任何 plaintext connection credential。除可视化 artifact 外，计算和大规模 latent storage 应留在远程环境。

Gate 1 的结论必须同时报告：

- generated combined / chunk PDiT；
- ground-truth encoded diagnostic state；
- valid queries、filter reasons、time mapping、confidence interval；
- 全部预注册失败类别，而非只给可视化成功案例；
- decoded-output consistency diagnostics。

若 external geometry GT、visibility rule 或 temporal mapping 未验证，则停止，不对 state addressability 下结论。

### Gate 2：MagicDriveDiT audit

只有官方 checkpoint、VAE、latent scaling/layout、geometry controls 和对齐数据可复现时才执行。以和 Gate 1 相同的 native-grid retrieval、外部 GT 和 metrics 判断：

- 是两类 generated video latent 的共享局限；
- 还是 OmniNWM 特有的 prediction/interface 问题；
- 或者证据尚不足以区分。

### Gate 3：DA3 / VGGT candidate controls

只在 Gate 1/2 已建立充分的 motivating evidence 后，按完全相同 query table 和 native retrieval protocol 测 DINOv2、DA3-Raw、DA3-DPT 与 VGGT。此 gate 不训练新的 NWM，也不训练 decoder 来让 candidate state 在 Claim 1 中获益。

若 DA3 / VGGT 没有稳定优势，正确结论是该几何 foundation-model route 没有被当前审计支持；不应继续把它包装为必然的 driving encoder。

### Deferred Gate 4：未来全视角 NWM 模块

只有 DA3 / VGGT 通过 Gate 3 后，再考虑：

```text
frozen geometry foundation-model encoder
→ compact state tokenizer（需单独验证对应性保留与成本）
→ rig / ray / future-trajectory conditioned generative transition model
→ predicted compact geometric state
```

这一步需重新验证 state dynamics、tokenizer、计算成本、decode 与驾驶效用，全部超出当前文件。

---

## 7. 后续实现可复用的工程位置

| 组件 | 未来用途 |
|---|---|
| [configs/inference/infer.py](OmniNWM-master/configs/inference/infer.py) | 官方 33-frame / 6-camera audit config 的基准；创建副本而不是修改默认文件 |
| [omninwm/utils/sampling.py](OmniNWM-master/omninwm/utils/sampling.py) | 在 `unpack` 后、decode 前添加 config-gated final PDiT capture hook |
| [tools/inference.py](OmniNWM-master/tools/inference.py) | 传递 sample identity、保存生成视频与 latent bundle 的入口 |
| [omninwm/utils/inference.py](OmniNWM-master/omninwm/utils/inference.py) | 扩展 per-sample raw artifact 写入；保留现有可视化视频流程 |
| [eval/consistency_suite.py](OmniNWM-master/eval/consistency_suite.py) | decoded-output consistency 的补充诊断，不承担 latent retrieval |
| `eval/predictive_state_localization.py`（未来新建） | 读取 native latent bundle、外部 GT 与 calibration，执行 Claim-1 geometry-localization audit |

未来 PDiT capture bundle 最少应包含：

```text
<save_dir>/video/<sample>.mp4
<save_dir>/video/raw/<sample>/
  predicted_pdit.pt
  predicted_pdit_rgb.pt       # 可选诊断
  predicted_pdit_depth.pt     # 可选诊断
  predicted_pdit_seg.pt       # 可选诊断
  depth.npy                   # decoded generated diagnostic
  calibration / pose / trajectory arrays
  source_paths.json
  meta.json
```

`meta.json` 至少记录：sample id、checkpoint/config、seed、view order、shape/dtype、combined/chunk layout、temporal reduction、bin-to-frame mapping、camera coordinate convention 与 source paths。

---

## 8. Claim-1 最终检查清单

在写出“某种 generated driving latent 缺乏或具备 geometry addressability”前，确认：

- [ ] 审计的是实际 generated final latent，而不是 initial noise、packed token、decoded RGB 或任意中间特征；
- [ ] ground-truth encoded state 与 generated state 分开分析；
- [ ] 所有 GT correspondence 来自外部 LiDAR / validated metric depth + calibration + ego pose，而非 generated depth 或 feature 自举；
- [ ] source/target 均通过 image bounds、front-facing、visibility、occlusion 与 depth-consistency 筛选；
- [ ] PDiT causal temporal bin 到 decoded-frame mapping 已文档化和验证；
- [ ] 所有 state 保留原生 channel、native spatial / temporal grid；没有 PCA、共享 channel 或 retrieval-stage forced resampling；
- [ ] 只在同一 state 内计算 cosine；跨模型只比较 external-GT localization outcomes；
- [ ] 主 PCK/error 在共同输入图像坐标报告，且 rank 在各自 native grid 报告；
- [ ] query sampling、scene split、seed、checkpoint/config 与失败样本均可复现；
- [ ] combined/chunk 仅作为同一 OmniNWM state 的诊断，不被当作独立 baseline；
- [ ] MagicDriveDiT 未能严格复现时明确标为 unavailable；
- [ ] decoded-output consistency 与 native-state addressability 结论分开报告；
- [ ] DA3/VGGT 仅被称为待检验 candidate controls，而不是预设答案；
- [ ] GLD 与 DVGT-2 仅作为外部动机，不被写成当前自动驾驶系统或 Claim-1 的证据；
- [ ] state profile（channels、tokens、bytes、extraction cost）与质量结果一起报告。

---

## 一句话总结

本工作的第一步不是宣称“DA3 比 VAE 好”，也不是重复 RAE-NWM 的“DINO 比 VAE 更线性可预测”。而是先审计：**现有多模态、几何条件的全视角 driving world model 所生成的 native latent，是否真的能在相机 rig 和时间轴上寻址同一物理世界。** 只有确认这种 state-level 缺口，并证明 DA3 / VGGT 在同一外部几何协议下更可寻址，才有理由进一步设计直接在预训练几何基础模型空间中生成的全视角导航世界模型。
