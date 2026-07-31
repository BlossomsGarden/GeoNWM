# GFM-DiT 训练计划

## Context

本文档说明如何把 OmniNWM 的“初始帧条件 + 未来相机编码”训练方式迁移到 GFM 特征空间。模型不再生成 Video VAE latent，而是在冻结 GFM 的特征空间中直接学习未来状态。

默认 GFM 为 **VGGT-1B**。第 2 节同时保留一种可替换的 GFM 状态定义。无论选择哪一种 GFM，DiT 的训练目标和最终输出都是供 DPT 解码器使用的四层原始 patch 特征按通道拼接后的完整状态。

---

## 1. 任务定义

基础任务以 `t=0` 的观测帧为条件，根据未来自车轨迹和相机参数，一次性联合生成后续 `H` 个时刻的 GFM 状态。默认训练设置为 `H=6`；单视角阶段使用 `CAM_FRONT`，多视角阶段使用同步六相机 rig。

```text
观测条件：t=0 的 GFM 状态
相机条件：未来各时刻、各视角的 normalized panoramic Plücker ray map
生成目标：未来 H 个时刻的完整 GFM 四层拼接状态
默认实现：VGGT-1B
默认视角数：V=6
```

六相机顺序固定为：

```text
CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT,
CAM_BACK_RIGHT, CAM_BACK, CAM_BACK_LEFT
```

多视角训练和推理必须把 `H × V` 个状态作为一个联合样本处理，不能拆成逐帧或逐相机的独立生成任务。

---

## 2. GFM 状态定义与特征预处理

GFM 在训练期间冻结。每个物理时刻独立提取供 DPT 解码使用的四层 patch 特征，删除 camera/register/其他非 patch token 后，只进行形状整理和通道拼接。

### 2.1 DA3-LARGE-1.1

取 DPT 解码器使用的层 `[11, 15, 19, 23]`。在 `cat_token=True` 时，每层 patch 特征为 2048 通道；按层号顺序沿通道维直接拼接：

```text
F_DA3 = cat(F_11, F_15, F_19, F_23, dim=channel)
F_DA3 : [B, V, P, 8192]
```

这里使用的是进入 DPT 解码器之前的四层原始 patch 特征，不使用 DPT 内部的投影、resize 或融合结果。

### 2.2 VGGT-1B

**VGGT-1B 是默认实现。** 取 DPT 解码器使用的聚合层 `[4, 11, 17, 23]`。VGGT-1B 的每层聚合特征由 frame token feature 与 global token feature 按通道拼接而成，因此每层为 `2 × 1024 = 2048` 通道。去掉 `patch_start_idx` 之前的 camera/register token 后，再按层号顺序沿通道维直接拼接：

```text
F_VGGT = cat(F_4, F_11, F_17, F_23, dim=channel)
F_VGGT : [B, V, P, 8192]
```

这里同样使用进入 DPT 解码器之前的四层原始 patch 特征，不使用 DPT 的 projection、resize、fusion 或 prediction head 输出。

两种 GFM 使用相同的状态契约：

- 不使用 VAE、tokenizer、PCA、跨层融合或空间下采样 latent；
- 不对四层特征做额外压缩、池化或统计变换；
- DiT 内部可以把 8192 通道线性投影到模型宽度，但 Flow Matching 的数据状态、监督目标和最终生成结果始终是完整的 8192 通道拼接特征。

---

## 3. 数据与条件组织

一个样本包含 `t=0` 观测和未来 `H` 个时刻。对每个时刻保存图像、相机内外参、自车位姿和冻结 GFM 提取出的四层拼接状态。多视角阶段要求同一时刻的六个相机完整且顺序固定。

初始帧条件采用与 OmniNWM 相同的时间对齐形式：条件张量在 `t=0` 位置写入观测 GFM 状态并置有效 mask，在未来位置填零。模型的 token 序列同时包含这个条件位置和所有未来 noisy state 位置，但训练损失只监督未来状态。

```text
cond_state[t=0] = F_0
cond_state[t>0] = 0
cond_mask[t=0]  = 1
cond_mask[t>0]  = 0
model_state[t=0] = 0
model_state[t>0] = X_tau
```

Stage 1 每个时刻只提取 `CAM_FRONT`；Stage 2 和 Stage 3 每个物理时刻以同步六视角作为 VGGT 输入并保留六个视角各自的 patch 状态。禁止把未来 RGB 或未来 clean GFM feature 放入条件分支。

---

## 4. 未来运动与相机几何条件

相机条件直接沿用 OmniNWM 的处理：根据未来自车轨迹以及各相机内外参，将每个视角的射线转换到以 `t=0` 前视相机为基准的统一 panoramic 坐标系，重新计算归一化 Plücker ray map，再按 GFM patch 网格与对应的未来状态 token 对齐。本文档不再展开独立的相机条件编码细节。

---

## 5. CausalRigFlowDiT 架构

### 5.1 输入与输出

`CausalRigFlowDiT` 接收由 `t=0` 零占位和未来 noisy GFM state 组成的状态序列，以及初始帧条件、未来 ray map 和 Flow Matching 时间 `τ`。输出时丢弃 `t=0` 占位，只返回与未来目标完全同形状的 velocity：

```text
v_theta(X_tau, tau | cond_state, cond_mask, ray_map)
    -> [B, H, V, P, 8192]
```

“Causal”表示 clean observation 只有 `t=0`；未来时刻之间可以联合注意，因为它们都是同时去噪的变量，而不是可作为条件读取的 future ground truth。

### 5.2 初始帧条件注入

初始帧不经过独立的上下文编码器，而是仿照 OmniNWM 形成与状态序列对齐的 condition tensor：

```text
model_state = cat(zeros_like(F_0), X_tau, dim=time)
state_token = state_in(model_state)
cond_token  = cond_in(cat(cond_mask, cond_state))
state_token = state_token + cond_token
```

`cond_in` 使用零初始化，使新增条件分支从不扰动主干开始学习。初始帧信息通过后续时空联合注意力传播到所有未来 token。

### 5.3 未来相机编码注入

未来 normalized panoramic Plücker ray map 不只作为 attention bias 或 AdaLN 向量，而是按照 OmniNWM 建立独立 ray-token 流：

1. ray map 按 GFM patch 网格打包，并由 `ray_in` 投影到与 state token 相同的模型宽度；
2. 双流 block 分别归一化和调制 state token、ray token，再拼接两者的 Q/K/V 做一次联合注意力；
3. 联合注意力的结果按 token 范围拆回两条流，各自完成 residual 和 MLP 更新；
4. 双流阶段结束后，将 ray token 与 state token 拼接送入单流 block；
5. 输出前移除 ray token 和 `t=0` state 占位，只将未来 state token 投影回完整 8192 通道 velocity。

因此，相机参数能够在每个双流 block 中直接参与未来状态建模，同时不会改变 GFM 状态本身的定义。

### 5.4 时空与多视角建模

模型保留空间、时间和跨视角建模。Stage 2 起，同一未来时刻的六个视角通过 panoramic cross-view attention 交换信息；不同未来时刻联合去噪。位置编码必须同时区分时间、视角和非方形 patch 坐标，padding token 不参与 attention 或 loss。

---

## 6. Flow Matching 训练目标

Flow Matching 直接定义在四层拼接后的原始 GFM 状态上。对完整未来联合状态采样同一个 `τ`：

```text
F_target = future full-channel GFM state
epsilon  ~ N(0, I)
X_tau    = (1 - tau) * F_target + tau * epsilon
u_tau    = epsilon - F_target

L_FM = mean_valid(
    ||v_theta(X_tau, tau | cond_state, cond_mask, ray_map) - u_tau||^2
)
```

损失只统计未来有效 patch。首版训练只使用 `L_FM`，不引入 RGB、depth、semantic、重投影损失或其他阶段化辅助目标。

---

## 7. 分阶段训练计划

训练课程沿用 OmniNWM 的渐进思路：先建立单视角相机控制能力，再扩展到六视角联合生成，最后通过变长度和变分辨率训练提高适应性。三个阶段连续加载上一阶段权重，GFM 始终冻结，状态定义和条件注入方式始终不变。

### Stage 1：单视角控制

- 使用 `CAM_FRONT`，固定 `H=6` 和基础分辨率；
- 输入 `t=0` 的 VGGT-1B 全通道状态以及未来前视相机 ray map；
- 输出未来六个时刻的完整 8192 通道 VGGT-1B 状态；
- 重点验证初始帧条件、ray-token 注入和基本时序生成能够稳定收敛。

参考 OmniNWM 的比例，首轮训练预算以约 10K iterations 为起点，最终停止点由验证集 Flow Matching loss 和生成状态误差决定。

### Stage 2：六视角联合生成

- 从 Stage 1 checkpoint 继续训练，保持固定 `H=6` 和基础分辨率；
- 将输入和目标扩展为同步六相机 rig，并启用 panoramic cross-view attention；
- 六个视角及六个未来时刻必须一次性联合生成，不能拆成 36 个独立样本；
- 每个未来视角使用与其内外参和未来自车位姿对应的 normalized panoramic ray token。

参考 OmniNWM，首轮扩展预算以约 3K iterations 为起点。该阶段只扩展视角维，不更换 GFM 目标，也不增加 RGB、depth 或 semantic 生成分支。

### Stage 3：变长度与变分辨率微调

- 从 Stage 2 checkpoint 继续训练，混合不同未来长度和不同输入分辨率；
- 基础设置保留 `H=6`，长序列可使用 `H=12`；高分辨率设置按 patch size 的整数倍扩展；
- 使用显式 valid mask 处理不同时间长度和 patch grid，仍对每个样本的全部有效视角联合去噪；
- 训练和推理继续使用同一套初始帧 condition tensor 与 ray-token 双流接口。

参考 OmniNWM，首轮微调预算以约 3K iterations 为起点。该阶段只改变长度和分辨率分布，最终输出仍是每个有效 future/view/patch 的完整 8192 通道 VGGT-1B 状态。

---

## 8. 实施与验收

实现只需要四个清晰边界：冻结 GFM 特征提取、初始帧 condition 构造、normalized panoramic ray-map 构造、CausalRigFlowDiT 与 Flow Matching 训练。三份 stage 配置仅改变视角数、序列长度、分辨率和训练步数，不复制模型实现。

首版实现必须通过以下检查：

1. VGGT-1B 确实读取 `[4, 11, 17, 23]`，每层 2048 通道，最终状态为 8192 通道；
2. 切换第 2.1 节的备用 GFM 时只替换冻结特征提取器，状态仍为四层直接拼接；
3. condition tensor 只有 `t=0` 包含 clean GFM feature，未来位置为零；
4. ray map 作为独立 token 流进入双流联合注意力，而不是只作为 bias 或 AdaLN 条件；
5. 输出移除 ray token 和 `t=0` 占位后恢复为 `[B,H,V,P,8192]`，并直接计算 `L_FM`；
6. Stage 1、2、3 分别对应单视角、六视角、变长度/变分辨率训练，不改变最终生成状态的定义。
