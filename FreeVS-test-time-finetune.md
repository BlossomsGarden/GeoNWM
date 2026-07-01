# FreeVS Test-Time Latent Uncertainty Noise Reinjection Research Brief

> 用途：这份文档是给后续科研/实现 agent 的任务提示，不是最终代码实现。当前阶段不要落代码；先围绕 FreeVS 推理时的 latent-space uncertainty region 与 noise reinjection 做可复现实验设计。

> 资深审稿人快速判断：这个想法有潜力，但现在最危险的写法是把它包装成 "latent 版 Laplacian + 一堆 proxy 融合"。这会被认为是工程堆料。更强的创新叙事是：在 FreeVS/SVD 的 denoising loop 内，不经 VAE decode、不增加主路径 UNet forward，用 clean-latent estimate 的时空收敛状态定位未稳定 latent token，并以 scheduler-aware 的方式局部 re-noising。实验必须证明三件事：mask 确实对应不稳定区域、局部重注入确实降低自回归/大视角偏移误差、额外 latency 在可接受范围内。

## 1. 用户目标

在 `test-new-mask/FreeVS-master/` 的 FreeVS 推理阶段，引入一种 test-time latent intervention：

1. 在 denoise step 过程中直接对 latent 计算不确定性区域。
2. 只在不确定性区域执行噪声重注入，帮助模型获得更好的去噪结果。
3. 思路参考 `test-new-mask/drivelaw-main/` 中 DriveLaW-Video 的 noise reinjection，但不要照搬其 pixel-space Laplacian mask，因为每步把 latent decode 到 pixel space 代价过高。
4. 重点研究 latent-space 的数学不确定性度量、mask 生成、scheduler-aware 噪声重注入策略，以及可替代方案。

这里的 "test-time finetune" 更准确地说是 "test-time sampling/refinement"：默认不更新模型权重，只改推理采样路径。

## 2. 当前代码观察

### FreeVS 推理链路

重点文件：

- `test-new-mask/FreeVS-master/diffusers/examples/freevs/inference_svd.py`
- `test-new-mask/FreeVS-master/diffusers/examples/freevs/src/pipelines/pipeline_stable_video_diffusion_custom.py`
- `test-new-mask/FreeVS-master/diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py`

当前 `inference_svd.py` 使用：

- `StableVideoDiffusionPipeline_convnb_multiframe.from_pretrained(...)`
- 默认 `num_inference_steps=25`
- `min_guidance_scale=2.0`, `max_guidance_scale=2.0`
- `noise_aug_strength=0.02`
- `generator = torch.manual_seed(42)`

`StableVideoDiffusionPipeline_convnb_multiframe.__call__` 的关键 denoising loop 位于 `pipeline_stable_video_diffusion_custom.py` 中。

FreeVS 使用 `EulerDiscreteScheduler`。本地 scheduler 的 `step()` 返回 `EulerDiscreteSchedulerOutput(prev_sample, pred_original_sample)`，但 FreeVS 目前只取 `.prev_sample`。后续 agent 可以在不改变 baseline 行为的前提下把返回值保存下来：

```python
step_output = self.scheduler.step(noise_pred, t, latents)
z0_hat = step_output.pred_original_sample
latents = step_output.prev_sample
```

这很重要：`pred_original_sample` 是当前 step 的 clean latent 估计，适合作为 latent-space uncertainty 的低成本输入，避免在 denoise loop 内调用 VAE decode。

注意：`pipeline_stable_video_diffusion_custom.py` 内有多个派生 pipeline 类和多段相似 denoising loop。当前入口使用的是 `StableVideoDiffusionPipeline_convnb_multiframe`，后续 agent 必须先用 shape log 或小规模 dry run 确认实际调用路径，避免改到未使用的类。

### DriveLaW noise reinjection 对照

重点文件：

- `test-new-mask/drivelaw-main/DriveLaW-Video/Infer/infer.py`
- `test-new-mask/drivelaw-main/DriveLaW-Video/Infer/README.md`
- `test-new-mask/drivelaw-main/DriveLaW-Video/Infer/diffusers/src/diffusers/pipelines/ltx/pipeline_ltx_condition.py`

DriveLaW 提供了这些参数：

- `--noise_reinjection_enabled`
- `--noise_reinjection_beta`
- `--noise_reinjection_sigma`
- `--noise_reinjection_steps`

其核心逻辑：

1. 对当前 packed latents 额外跑一次 transformer，估计 clean latent。
2. 将 clean latent unpack 并 VAE decode 到 pixel space。
3. 对 pixel frames 转灰度后做离散 Laplacian。
4. 用 `threshold = beta * std(abs(Laplacian(frame)))` 得到高频 mask。
5. 将 pixel mask nearest downsample 到 latent resolution。
6. 在原始 latent 上执行 `latent + sigma_prime * mask * noise`。
7. 再进入正常 denoising step。

DriveLaW README 也明确说：noise reinjection 主要缓解高速驾驶视频中的局部运动伪影和 ghosting，并不根本解决架构问题，也不保证通用视频生成质量大幅提升。

对 FreeVS 的启发：

- 可借鉴 "只在早期 steps 作用"、"只改 mask 区域"、"重注入强度可调"。
- 不建议照搬 "VAE decode + pixel Laplacian"，因为每个 denoise step 解码视频 frames 的成本太高。
- 也不建议把 "高频边缘" 等价成 "不确定性"。边缘可能稳定、清晰；真正需要处理的是 denoising trajectory 不稳定、condition disagreement、时序不一致或 score variance 高的区域。

## 3. 文献与技术依据

后续 agent 需要优先读这些材料，并在实验报告中记录每个方法对应的证据来源：


- DiffEdit: Couairon et al., "DiffEdit: Diffusion-based semantic image editing with mask guidance", arXiv:2210.11427, https://arxiv.org/abs/2210.11427  
  依据：可以用 denoiser 对不同条件的 noise prediction 差异生成 mask。这支持 FreeVS 中用 `noise_pred_cond - noise_pred_uncond` 作为一种 condition-disagreement signal。

- Uncertainty-guided diffusion sampling: "Diffusion Model Guided Sampling with Pixel-Wise Aleatoric Uncertainty Estimation", arXiv:2412.00205, https://arxiv.org/abs/2412.00205  
  依据：可用 denoising score/noise prediction 的方差估计不确定性，再用不确定性指导采样。FreeVS 中可把 pixel-wise 思路迁移为 latent-token/latent-cell variance。

- SDEdit: Meng et al., "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations", arXiv:2108.01073, https://arxiv.org/abs/2108.01073  
  依据：对已有表示加噪再去噪可以作为 test-time refinement 手段。

- RePaint: Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion Probabilistic Models", arXiv:2201.09865, https://arxiv.org/abs/2201.09865  
  依据：局部/条件区域的 resampling 和 re-noising 可以改善扩散模型在受约束区域的生成。

- FreeInit: "FreeInit: Bridging Initialization Gap in Video Diffusion Models", arXiv:2312.07537, https://arxiv.org/abs/2312.07537  
  依据：视频扩散可在 test-time 直接操作 latent initialization/noise 来提升质量和时序稳定。

- FreeNoise: "FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling", arXiv:2310.15169, https://arxiv.org/abs/2310.15169  
  依据：视频扩散的 noise schedule / latent noise 操作可以在不训练的情况下影响长视频一致性。

- CoNo: "Consistency Noise Injection for Tuning-free Long Video Diffusion", arXiv:2406.05082, https://arxiv.org/abs/2406.05082  
  依据：进一步支持在视频扩散中研究 consistency-aware noise injection，但 FreeVS 第一版仍应优先做局部 uncertainty mask，避免全局注入导致闪烁。

- Classifier-Free Diffusion Guidance: Ho and Salimans, arXiv:2207.12598, https://arxiv.org/abs/2207.12598  
  依据：FreeVS denoising loop 已经产生 uncond/cond 两类 prediction；它们的差异可以被视作条件敏感度或局部不稳定代理。

- Common Diffusion Noise Schedules and Sample Steps are Flawed: Lin et al., arXiv:2305.08891, https://arxiv.org/abs/2305.08891  
  依据：采样步、噪声尺度和 CFG rescale 会显著影响结果；设计 `rho_i` 与 `sigma_i` schedule 时必须保持 scheduler-aware。

## 4. 推荐主方案：Geometry-Informed Latent Uncertainty + Scheduler-Consistent Re-sampling

RGB 图像的边缘算子可以解释为像素空间结构变化，但 FreeVS 里的 video latent 是学习到的多通道表示，通道没有 RGB 那种可解释物理含义。真正值得强调的创新点应是：在不 decode、不额外 UNet forward 的前提下，用 denoising trajectory 中 clean-latent estimate 的局部收敛不稳定性来定义 video latent uncertainty；同时既然 conditioning 显式包含逐帧点云/几何，就再加一个 geometry-informed prior，专门描述哪里信息密度不足、哪里需要更多生成自由度，并只对这些区域做 scheduler-consistent masked re-sampling。

因此第一版主方案要尽量克制：一个 timestep-adaptive latent 信号，一个几何先验，一个条件敏感度诊断项。不要把这些东西再做一次经验拼盘。

### 4.1 记号

令：

- `z_i`: 第 `i` 个 denoise step 开始时的可演化 noisy latent，形状约为 `[B, T, C_z, H, W]`。
- `image_latents`: 由输入 pseudo images / condition images 编码得到的条件 latent。FreeVS 当前实现会把 `z_i` 与 `image_latents` 在 channel 维拼接后送入 UNet，因此 "16 通道 latent" 需要拆开理解：uncertainty mask 应作用在可演化的 `z_i` / `z0_hat_i` 上，而不是作用在拼接后的条件通道上。
- `t_i`: 当前 scheduler timestep。
- `sigma_i`: 当前 timestep 对应的噪声尺度。
- `eps_u`, `eps_c`: CFG split 得到的 unconditional / conditional noise prediction。
- `eps_g = eps_u + guidance_scale * (eps_c - eps_u)`: guided prediction。
- `step_output = scheduler.step(eps_g, t_i, z_i)`。
- `z0_hat_i = step_output.pred_original_sample`: 当前 step 的 clean latent 估计。
- `z_next_det = step_output.prev_sample`: baseline Euler step 产生的下一个 latent。
- `G_geo_i`: 几何可靠性分数。
- `M_geo_i`: 几何 mask。
- `E_i`: per-location EMA for timestep 修复。
- `warmup_steps`: 前若干步只积累统计量，不做 reinjection。
- `tau_traj`, `tau_geo`: latent / geometry 的阈值。

最终生成一个 latent mask：

```text
M_i in {0, 1}^{B x T x 1 x H x W}
```

mask 为 1 的区域是当前 step 认为尚未收敛、需要局部 stochastic refinement 的 latent cells。它不是 RGB 边缘 mask，也不是对 16 个通道分别做边缘检测后投票。

### 4.2 核心 uncertainty 定义

我们不再保留二阶时序差分。对驾驶视频来说，它很容易把合理运动和模型失效混在一起，审稿人会直接问你是不是在拿 motion 当 uncertainty。

#### A. Geometry-informed reliability prior

既然 conditioning 显式包含逐帧点云 / 几何，就不要只从 latent 里猜。先把 frame-wise point cloud 投影到目标视角，构造几何可靠性图：

```text
G_geo_i = w_d * (1 - D_density_i)
        + w_g * D_depth_gap_i
        + w_o * D_occl_i
```

其中 `D_density` 是局部点密度，`D_depth_gap` 是深度断裂或跳变，`D_occl` 是遮挡 / 缺失指示。`M_geo_i` 不是从图像里找问题，而是从 3D 关系推导哪里信息密度不足，所以它对远处行人、遮挡边界、平坦投影伪影和深度估计不稳区域更有物理意义。

```text
M_geo_i = threshold(G_geo_i, tau_geo)
```

#### B. Timestep-adaptive latent trajectory uncertainty

`U_traj` 只在修复 timestep 混淆后才有资格当主信号。先对 channel 做 whitening，再对每个空间位置维护自己的 EMA：

```text
Delta_i = z0_hat_i - stopgrad(z0_hat_{i-1})
s_{i,c} = 1.4826 * MAD_{T,H,W}(Delta_i[:, :, c, :, :]) + eps
U_raw_i = sqrt(mean_c((Delta_i[:, :, c, :, :] / s_{i,c})^2))
E_i = beta * E_{i-1} + (1 - beta) * U_raw_i
U_traj_i = log((U_raw_i + eps) / (E_i + eps))
```

`warmup_steps` 默认 2，可在 `0 / 2 / 4` 间小范围搜索；warmup 内只更新 `E_i` 和 `M_geo_i`，不做 reinjection。`U_raw_i` 只保留作诊断对照，`U_traj_i` 才是主方法候选。`A0-raw` 和 `A0-ema` 应成对报告，详细的 timestep 混淆修复可参见 `U_traj-timestep-confounding-fix.md`。这样做的好处是：早期步的全局噪声尺度被 EMA 吸收，late step 才会真正暴露“哪里收敛慢”。

#### C. Condition-disagreement as audit, not primary uncertainty

受 DiffEdit 启发，利用 CFG 中已有的 split：

```text
U_cfg_i = RMS_C(eps_c - eps_u)
```

这不是纯粹的不确定性，而是条件敏感度。高 `U_cfg` 可能表示该区域强依赖输入条件，也可能表示条件与当前 latent 发生冲突。它只适合做诊断或 veto，不适合做正向融合项。

- 诊断项：报告被 reinjection 区域的 `U_cfg` 分布，判断是否总在强条件区域扰动。
- 风险门控：若 `U_cfg` 极高但 `U_traj` 和 `M_geo` 都不高，默认不加噪。
- 消融对照：单独测试 `U_cfg` mask，证明“条件敏感”不等于“需要 re-sampling”。

FreeVS 的 `do_classifier_free_guidance` 逻辑与标准 diffusers 略有不同，后续 agent 必须先确认 `noise_pred.chunk(2)` 的 batch/shape 真实可用。

#### D. Denoising update magnitude as logging only

如果某区域每步更新幅度异常大，可以作为补充信号：

```text
U_update_i = RMS_C(z_next_det - z_i) / (abs(sigma_i - sigma_next) + eps)
```

该项对 scheduler scale 敏感，不建议进入第一版 mask 生成。保留为 debug logging 即可，用来解释失败 case。

### 4.3 Mask construction, warmup, and no-grad

所有 scores 和 masks 都在 `torch.no_grad()` 下计算，并在进入 reinjection 前 `detach()`；mask 只是 inference-time routing signal，不参与梯度。`warmup_steps` 默认设为 2，前几步只更新 EMA 和几何统计，不做 reinjection。

主方法不再做固定权重融合，而是把两个 proposal mask 合并：

```text
M_lat_i = U_traj_i > tau_traj
M_geo_i = G_geo_i > tau_geo
M_i = M_lat_i OR M_geo_i
M_i = M_i AND NOT G_cfg_veto_i
```

`tau_traj` 的自然含义是 log-ratio 是否打破稳态，`tau_geo` 则表示几何可靠性不足。若 union 后 coverage 超过预算，就按 `U_traj` 与 `G_geo` 的联合分数截断到 top-k。`A0-raw`、`A0-ema`、`A_geo`、`A_mix` 都要作为独立候选保留。

如果需要按目标 coverage 统一比较不同方法，`tau_traj` 和 `tau_geo` 也可以分别用 quantile / MAD 做阈值校准；这时 `A0-q` / `A0-MAD` 只是同一主信号的不同 calibration 版本，不是新的信号源。

### 4.4 Scheduler-consistent masked resampling

优先研究 masked resampling，而不是单纯的噪声混入。这里用 one-step RePaint 近似：先把 `z0_hat_i` 按 scheduler 对齐地重新加噪，再只在 mask 区域替换下一步 latent。

```text
z_next_det = step_output.prev_sample
z0_hat_i = step_output.pred_original_sample
eta = randn_like(z0_hat_i)
z_reinject = scheduler.add_noise(z0_hat_i.detach(), eta, t_next)
z_next = where(M_i, z_reinject, z_next_det)
```

如果 `scheduler.add_noise` 在当前 fork 里不可靠，就退回 `z0_hat_i + sigma_next * eta` 作为显式对照，但它不应成为主方法。`mask` 进入替换前应先 `detach()`，避免任何梯度语义混进来。

这样做的优点是：

- 比 `z0_hat_i + sigma * eta` 更贴近 scheduler 的噪声轨迹。
- 不需要 decode。
- 不需要额外 UNet forward。
- 比“先 deterministic 再 stochastic 混合”更像局部重采样，而不是单纯抖动。

### 4.5 Pre-step additive reinjection 作为轻量对照

DriveLaW 风格的前置扰动只保留为对照：

```text
z_i_prime = z_i + rho_i * sigma_i * M_{i-1} * eta
```

它实现更简单，但 mask 滞后一拍，且更容易放大 CFG 不稳定，所以不作为主路线。

## 5. 备选研究路线

### Option B: MC Score Variance Mask

这是更接近 uncertainty estimation 的方法，但计算更贵。

核心思想：

1. 对当前 step 的 latent 生成 `K` 个轻微扰动版本，或从 `z0_hat_i` re-noise 到当前 `sigma_i`。
2. 对这 `K` 个版本分别跑 UNet，得到 `eps_g^{(1..K)}`。
3. 计算每个 latent cell 的 prediction variance：

```text
U_mc_i = Var_K(eps_g^{(k)})
```

初始配置：

- `K=3`，只在前 `3` 个 steps 或每隔 `4` 个 steps 运行一次。
- 其他 steps 复用最近一次 mask，或退回主方案 `U_traj`。

优点：最像真正的 epistemic/aleatoric uncertainty proxy。

缺点：额外 `K` 次 UNet forward，成本高。FreeVS 视频 UNet 本来就贵，所以不推荐作为第一落地方案，但适合作为 "upper-bound ablation"。

### Option C: DiffEdit-style Condition Disagreement Mask

只使用已有 CFG split：

```text
M_i = threshold(RMS_C(eps_c - eps_u))
```

这是几乎零成本的 mask。建议作为 baseline，而非最终方案。

可扩展项：如果 FreeVS 的 `image_latents` 与 `z0_hat_i` 在同一 latent space/shape 上可对齐，可以试：

```text
U_cond_lat_i = RMS_C(z0_hat_i - image_latents_condition)
```

但要非常谨慎：`image_latents` 是 pseudo image / conditioning latent，不是目标 GT；差异可能代表合理生成自由度，不一定是不确定性。

### Option D: Latent Band-Energy / Wavelet Mask

如果需要一个 "不像 Laplacian 但仍关注局部结构" 的高频 baseline，可以用：

- Difference of Gaussian in latent space：`z0_hat - avg_pool(z0_hat)`
- Haar wavelet high-pass energy
- FFT band energy on spatial or temporal dimensions

例如：

```text
U_band_i = RMS_C(z0_hat_i - avg_pool2d(z0_hat_i, kernel=5))
```

这仍然偏向 high-frequency，不等价于 uncertainty。它适合做对照，验证 DriveLaW 的 Laplacian 思路是否真的需要 pixel decode。

### Option E: Local Guidance Damping Instead of Noise Reinjection

若不确定性区域主要来自 CFG 过强，可以不加噪，而是局部降低 guidance：

```text
eps_g_local =
  eps_u + guidance_scale * (1 - kappa * M_i) * (eps_c - eps_u)
```

初始 `kappa=0.25 / 0.5`。

优点：无随机噪声，可能更稳定。

风险：可能削弱条件遵循，尤其是道路结构、物体布局等本来就需要强条件约束的区域。

## 6. 评价指标与验收

测试场景收紧为 `REMOTE_SERVER.md` 所示远程服务器 project_dir 下的 `/data/wlh/FreeVS/waymo_process` 场景，起点统一为第 `145` 帧。所有方法使用同一场景、同一起点、同一输出设置，只在 Method 对应策略上变化。

每个方法只记录以下数值。记录 inference latency，是为了判断 "以时间换质量" 是否值得。

| Method | Quality rank | Total time | Denoise time | Peak VRAM | seed |
|---|---:|---:|---:|---:|---|
| Baseline |  |  |  |  |  |
| R0: random mask with same coverage |  |  |  |  |  |
| A0-raw: raw channel-whitened `U_traj` + post-step |  |  |  |  |  |
| A0-ema: EMA-normalized `U_traj` + post-step |  |  |  |  |  |
| A0-q: channel-whitened `U_traj` + quantile mask + post-step |  |  |  |  |  |
| A0-MAD: channel-whitened `U_traj` + MAD mask + post-step |  |  |  |  |  |
| A_geo: geometry-informed `M_geo` only + post-step |  |  |  |  |  |
| A_mix: `U_traj` OR `M_geo` + post-step |  |  |  |  |  |
| A1-q: A0-q + temporal gate |  |  |  |  |  |
| A1-MAD: A0-MAD + temporal gate |  |  |  |  |  |
| A2-q: A1-q + CFG veto/audit |  |  |  |  |  |
| A2-MAD: A1-MAD + CFG veto/audit |  |  |  |  |  |
| B1: `U_cfg` only / DiffEdit-style condition disagreement |  |  |  |  |  |
| B2: `U_cond_lat` if `image_latents` align with `z0_hat_i` |  |  |  |  |  |
| C1: latent Difference-of-Gaussian band energy |  |  |  |  |  |
| C2: latent Haar / FFT high-pass band energy |  |  |  |  |  |
| D1: old weighted fusion |  |  |  |  |  |
| E1: MC variance, K=3 |  |  |  |  |  |
| F0-add: best mask + post-step additive noise baseline |  |  |  |  |  |
| F1: best A + pre-step additive reinjection |  |  |  |  |  |
| G1: best mask + local guidance damping |  |  |  |  |  |

每一行方法都要做下面这些可视化结果，均使用新轨迹下的前三 camera views：

1. 原轨迹推理验收：在原始轨迹或与训练/输入条件最接近的轨迹上推理，观察行人、标示牌、车辆轮廓、车道线是否更精确。重点不是整体观感，而是小目标边界、文字/标牌结构、远处行人的稳定性是否相对 baseline 改善。
2. 偏离轨迹 30 帧自回归验收：初始帧从原帧开始；之后每个 chunk 都以上一个 chunk 的末帧作为下一 chunk 的起点，连续生成 30 帧。比较 baseline 与方法在行人/标牌漂移、车道线断裂、局部 ghosting、背景纹理漂移上的累积误差，验证是否降低 autoregressive error。
3. 大视角偏移验收：对同一 scene 做 `3m / 6m / 9m` 视角偏移，前三 camera views 分别输出对比。检查物体身份是否保持、车道线是否连续、路面与静态物体是否稳定、遮挡边界是否出现撕裂或随机纹理。偏移越大，越要警惕方法只是锐化局部而破坏几何一致性。

## 7. 期望结果

成功标准保持简单：A0/A1/A2 的可视化结果应优于 baseline 和 high-frequency/CFG 对照，尤其是在行人、标示牌、车道线、自回归漂移和大视角偏移稳定性上。若质量提升不明显但 `Total time`、`Denoise time` 或 `Peak VRAM` 明显变差，则不应继续作为主方案。

如果 `U_traj` 系列不能稳定优于 `U_cfg` only 或 latent band-energy / wavelet，只能把它写成 trajectory-change heuristic，不能声称是可靠 uncertainty estimation。

## 8. 后续 agent 执行提示

请按以下顺序执行科研任务：

1. 只读代码确认当前入口：`inference_svd.py` 是否仍调用 `StableVideoDiffusionPipeline_convnb_multiframe`。
2. 在不改变输出的情况下，做一次 shape tracing，记录：
   - `latents.shape`
   - `image_latents.shape`
   - `noise_pred.shape`
   - `noise_pred_uncond/cond.shape`
   - `step_output.pred_original_sample.shape`
   - `scheduler.sigmas` 与 `timesteps` 对齐方式
3. 实现前先写小张量单元测试，验证：
   - uncertainty map 输出 `[B, T, H, W]`
   - mask 输出 `[B, T, 1, H, W]`
   - mask coverage 可控
   - scheduler-aware reinjection 不改变 shape/dtype/device
4. 第6节的表格里每行对应一个文件夹吧，方便管理
5. 默认关闭该功能，保证 baseline 完全不变。
6. debug 输出不要默认保存视频级大 tensor；只在 flag 打开时保存压缩 `.npz` 或少量 step 的 mask stats。
7. 实验报告必须包含 baseline 对照、超参表、第 6 节记录表、固定 seed 复现说明、CLI参数、三类可视化验收。
