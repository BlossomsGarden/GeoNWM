<!-- 明白。现在我落了一个消融实验规划文档 [ablation.md](c:/Users/86187/Desktop/4/test-anchor/ablation.md) ，并且把原来的 Sync-Diffusion 重命名为 Sync-Diffusion-Base、Async-Diffusion重命名为Async-Diffusion-Full。关注这个文档里的S1、S1+mask、S1+async这三项，需要你依次完成的任务是——1、修改 [Sync-Diffusion-Base](c:/Users/86187/Desktop/4/test-anchor/Sync-Diffusion-Base/) 的代码，不做anchor帧，即首帧并不替换，而是像DiST-S一样通过额外的条件注入进去，使得它完美符合 [ablation.md](c:/Users/86187/Desktop/4/test-anchor/ablation.md) 里面S1的要求；2、基于前面已经做过的修改，复制过来并创建一个@Sync-Diffusion-mask 文件夹，向里面加入 mask的逻辑（*对原来只有 16ch的输入多扩展一个通道放 max-pool downsample后的mask，即17个通道，输出也仍为8个通道*），并且增加mask Loss，具体mask的处理、通道扩充、mask loss的计算代码你可以在 [Async-Diffusion-Full](c:/Users/86187/Desktop/4/test-anchor/Async-Diffusion-Full/) 文件夹里面找找（最终 loss 为：L = λ_rgb L_rgb + λ_depth L_depth + 原有 mask-region loss），最终目的是使得@Sync-Diffusion-mask 文件夹的代码完美符合 [ablation.md](c:/Users/86187/Desktop/4/test-anchor/ablation.md) 里面S1+mask的要求；3、执行完前面两个修改后，现在专注于达到 S1+async的要求。具体为：参考 [Async-Diffusion-Full](c:/Users/86187/Desktop/4/test-anchor/Async-Diffusion-Full/) 的代码，不做anchor帧，即首帧并不替换，而是像DiST-S一样通过额外的条件注入进去，并且没有mask和mask loss。最终形成一个 @Async-Diffusion-Base 文件夹。——现在你在执行前先仔细深入思考并针对我没有说清楚的地方或者细节向我提问 -->


<!-- 现在继续修改吧：1、统一都使用 --ema，mixed_precision fp16（比no更快更省显存），不开 --gradient_checkpointing；2、Sync-mask 应当直接删除 --use_mask_channel 这个命令行参数，因为这个文件夹底下的代码目的就是要有mask 这个通道，所以是必定传入并开启的，不要留一个冗余的 命令行参数；3、修改max_train_steps=60k，checkpointing_steps=2500（执行eval），然后删除checkpoints_total_limit这个命令行参数的逻辑；4、现在如果设置 batch_size>1 会出现bug，例如ConvNeXt模块部分存在写死batch_size1的逻辑，可以考虑改成把[B,N,C,H,W] reshape 成 [B*N,C,H,W]喂入ConvNeXt，squeeze就删除，其他的repeat或者别的变量也要都尽可能扫一遍在保证维度同时保证各batch的数据彼此相对应不能混乱，因为我希望增加batchsize训练，不想让这个bug一直存在 -->


以下出现的路径全都是本目录文件夹下：

S0：复现原DiST-S（no SCC）

S1：将S0替换成我的在线offset训练机制，参考代码 @Sync-Diffusion
S1+mask
S1+async   ——  额外做一个折线图展示S1+async的Δ的选定对FVD的影响 
S1+双anchor

Ours（S1+mask+async+双anchor）

其中：
S0 参考代码 @DiST-4D-main
S1+双anchor 应该是在现有的 @Sync-Diffusion-Base 修改后，再基于它增加首尾双anchor帧，且anchor帧不参与loss计算
Ours 参考现有代码 @Async-Diffusion-Full 但需要做一个小改动：首尾双anchor帧，且anchor帧不参与loss计算


那么如何做双anchor帧的注入？后面再说先关注S1集合。

 

# SVD 实现细节

原SVD 8 个输入通道 4ch noisy RGB video latent + 4ch image condition latent，其中后者为image->VAE encoder，再复制frames份形成伪video latent, 4 个输出通道。



# S0 实现细节（DiST-4D-main）

内部训练 span 是 6 帧，


Unet 16 个输入通道 4ch noisy target RGB latent + 4ch noisy target depth latent + 4ch warpped RGB condition latent + 4ch warpped depth condition latent，其中后两个是RGB-D条件直接通过 ConvNeXt supertiny 风格的 LayoutCondEncoder 编码得到，8个输出通道。训练不是直接拿 model_pred 和 noise 做 loss，而是 EDM/SVD 风格的 preconditioning。
lambda_rgb=1, lambda_depth=1,

此外，不是“完整 RGB 图片/深度图片各一张输入clip”，而是仅有单张完整RGB帧做提示。

1.数据集怎么取样的？
2.获得clip然后Repeat后如何与前面提到的16channel交互？


# S1 实现细节（Sync-Diffusion-Base）

nuscenes (/data2/DATA/AutoDrive/nuscenes/v1.0-trainval) + lidar 标定深度 (/data/wlh/MoGe-2/outputs/nuscenes_moge2_depth_256x704_fp16/align_depth)。具体采样逻辑为训练数据采样以预处理好的 anchor keyframe 表（远程服务器/data4/DATA/wlh/Difix3D+/nuscenes_train_samples.csv + nuscenes_val_samples.csv）为基本采样单元，场景内每个chunk包含6帧，起点为 chunk_start_id: 2, 8, 14, 20, 26, ..., 如果最后一个chunk不满6帧则直接丢弃。每个chunk的额外 CLIP image condition 就使用该chunk首帧的前面第2帧，例如起点为2，则取0的图片为CLIP image condition。

不全量提前存 14 种 offset，而采用offset在线生成的方案，DataLoader 开 num_workers，让 CPU 几何和 GPU 训练重叠。训练数据包括六视角相机，只是会分别输入训练/推理。对取出6帧chunk执行预设视角偏移的 offset = [逐帧向左/右渐进式地施加偏移量0~2米,2~4米,4~6米；逐帧向左/右施加固定偏移量1米,2米,4米,6米]。视角偏移的执行流包括：通过恰当的resize_crop逻辑将rgb帧与深度（理论上应该是448x768）的分辨率对齐，并同时修正相机内参，然后将摄像头进行对应逐帧的offset，获得虚拟新视角下的rgb帧以及可见区域的逐像素深度值（需要在 shifted view 里做近邻深度反穿透过滤）。然后根据这个RGB-D结果逐帧又反投影为点云再用原数据集的相机投影获得原相机视角下的存在不可见区域的Rgb、depth、不可见区域mask。于是联合最初的完整RGB、depth形成监督数据。mask可视化时白色标识不可见区域和失真区域。

具体采样逻辑为 __getitem__ 的 idx 不直接表示原始 frame_id，而被拆分为 chunk_start_id 与 offset_type：offset_type 按当前阶段的 offset pool 做 round-robin，anchor_id 通过每个 epoch 重新打乱的 anchor permutation 轮转采样，保证在一个 virtual epoch 内每个 anchor 与每种 offset 尽量等概率配对。前 50% training steps 只使用 progressive left/right 0->2m、2->4m 以及 fixed left/right 1m、2m、4m，共 10 类 offset，让模型先学习稳定补洞、anchor 条件使用和 chunk 内时序一致性；后 50% training steps 切换到全 14 类 offset，即 progressive left/right 0->2m、2->4m、4->6m 与 fixed left/right 1m、2m、4m、6m。

Depth的处理沿用 S0 的方案，不过由于深度信息来源不同，本方案中会首先将天空区域设置为100（DiST-S原方案是将语义天空部分设置为100），然后重投影失败的区域设置为0，完成 offset 阶段后再将 target depth 与 warped depth condition 分别 clip 到 [0.5, 100]，除以 100 归一化，并 repeat 成 3 通道作为网络输入表示。


# S1+mask 实现细节（Sync-Diffusion-Mask）

数据处理与采样逻辑同 S1，只是 dataset 取数据时会额外返回 mask_values，然后 UNet input 从 16ch 改成 17ch，多出的一个维度为该 mask 通过 max-pool 下采样到 latent 尺度，用于同时给 warpped RGB/Depth Video Latent 标识需要补全的区域。
总loss也会再增加与mask区域有关的项，一个chunk内的全部6帧都参与loss计算。
L = λ_rgb L_rgb + λ_depth L_depth + λ_mask_rgb L_mask_rgb + λ_mask_depth L_mask_depth
默认权重沿用 lambda_rgb=1, lambda_depth=1, lambda_mask_rgb=2, lambda_mask_depth=3


# S1+async 实现细节（Async-Diffusion-Base）

S1 的同步 RGB-D 扩散并不是简单地“同时生成 RGB 和 Depth 但效果不够好”，更核心的问题在于它把两个性质不同的预测目标放进了同一个去噪进度中。共享 timestep/sigma 实际上隐含了三个假设：RGB 与 Depth 在相同噪声水平下具有相近的恢复难度；两个模态在去噪过程中应当以相同速度从噪声转向结构；某一时刻的 RGB 与 Depth 中间状态对彼此都是同等可靠的条件。对于大视角 RGB-D 补全，这三个假设都偏强。Depth 在本任务中主要承载场景几何、遮挡关系和新视角可见性，其错误往往表现为低频结构漂移、前后景深度层次混乱和洞区边界不稳定；RGB 则主要承载纹理、材质、光照和细粒度外观，其错误更多表现为高频模糊、局部纹理幻觉和跨帧闪烁。二者既相关，又不是同一种不确定性。尤其在大视角 offset 的补全区域内，warped RGB/Depth condition 本身带有空洞、反投影噪声和遮挡边界误差，RGB 的可观测信息比 Depth 更稀疏、更局部，也更依赖已有几何支架。如果训练时仍使用同一个 timestep/sigma，网络在每个噪声水平都被要求同时恢复几何布局和外观纹理，容易形成两类冲突：一方面，RGB 分支会在几何尚未稳定时过早补纹理，导致纹理被错误深度边界牵引；另一方面，Depth 分支会被 RGB 纹理噪声和局部外观损失拖拽，降低其作为全局结构先验的稳定性。

因此，S1+async 的动机不是把同步生成机械地改成“先 Depth 后 RGB”，而是把 RGB-D 生成重新表述为一个有偏序关系的联合去噪问题：Depth 先收敛到较干净的几何状态，为遮挡边界、空洞区域和跨帧结构提供低频约束；RGB 随后在更可靠的几何状态附近细化纹理，从而减少大视角补全时的纹理漂移和几何-外观不一致。这一假设与 /paper/Semantics Lead the Way.txt 的启发相关，但本任务的迁移点更具体：原论文使用语义特征作为中间引导，而这里将显式生成的 Depth 作为可监督、可评估、可参与最终输出的几何模态。Depth 不是用后即弃的隐式条件，而是同时承担中间结构约束和最终预测目标，因此需要训练噪声、time embedding、preconditioning、loss weighting 和采样更新都按模态拆分，避免只在推理阶段人为错开 denoising 而造成 train-test mismatch。

数据处理与采样逻辑同 S1。训练时不再采一个共享 timestep/sigma，而是采一个全局进度 u ~ Uniform(0, 1 + Δ)，tau_depth = min(u, 1)，tau_rgb = max(u - Δ, 0)，其中 tau=0 表示最 noisy，tau=1 表示最 clean。随后将 tau 映射到 SVD/EDM 的 sigma schedule，并对 RGB 与 Depth 分别加噪（sigma(tau) 使用 log-normal 的 inverse-CDF，并 clamp tau 到 [eps, 1-eps]）。这样 Δ=0 时，sigma 分布严格退化成同步训练分布，可用于验证异步实现本身没有改变 S1 的基本训练目标。

模型输入仍为 16 channels，输出仍为 8 channels，并按通道拆分为 RGB prediction 和 Depth prediction。训练 loss 沿用原 DiST-S/SVD 的 EDM/v-pred 目标定义，但所有依赖 sigma 的 preconditioning 和 loss weighting 都按模态分别计算：RGB 分支使用 sigma_rgb 的 c_in/c_skip/c_out/c_noise/weight，Depth 分支使用 sigma_depth 的对应项。最终 loss 为：L = λ_rgb L_rgb + λ_depth L_depth，其中 lambda_rgb=1, lambda_depth=1 保持与 S1 一致。

模型侧需要把原 single timestep embedder 替换为模态感知的双时间条件。具体采用 emb_rgb 和 emb_depth 分别编码两个模态的噪声进度，再 concat 后通过 linear fuse 回原 hidden 维度。初始化时从原 time_embedding 拷贝参数，fuse 初始化成平均或等价线性投影，使 Δ=0 或训练初期尽量接近原同步模型，降低架构变化本身带来的干扰。

采样时 loop 也不能再用一个 scheduler step 同时更新所有 8 个 latent channel，而要按模态 split 更新并采用 Depth-leading schedule：u ∈ [0, Δ) 时只更新 Depth，RGB 保持初始高噪声状态；u ∈ [Δ, 1) 时 Depth 和 RGB 同时更新，但 Depth 的 denoising 进度始终领先 RGB；u ∈ [1, 1+Δ] 时 Depth 固定在 clean/near-clean 状态，只继续更新 RGB。该方案的推荐默认超参为 Δ=0.3，后续补充 Δ∈{0, 0.1, 0.3, 0.5} 的 ablation，并额外绘制 Δ 对 FVD 的影响曲线。若 Δ 过小，几何先验尚不足以明显约束 RGB；若 Δ 过大，RGB 可用的联合细化窗口变短，可能削弱纹理质量和 RGB-D 的后期协同。因此 Δ ablation 不只是调参，而是检验“几何领先多少最有利于视频质量”的核心证据。

额外消融方案：Δ=0 版本。相比于 S1，它只把模型架构改成 Async-capable，但不引入实际的模态去噪时序差异，用于区分“架构拆分带来的收益”和“Depth-leading 去噪顺序带来的收益”。


# S1+双anchor


# Ours（Async-Diffusion-Full）




# 超参数

统一都使用 --ema，mixed_precision fp16（比no更快更省显存），不开 --gradient_checkpointing
train_batch_size	1 / device
gradient_accumulation_steps	4
max_train_steps=60k，checkpointing_steps=2500（执行eval）
learning_rate	5e-05
LR warmup 500
lr_scheduler	constant
checkpointing_steps	2500
checkpoints_total_limit	20
dataloader_num_workers	6
nframes	6
video_width x video_height	768 x 448
max_grad_norm	1
offset_mode	curriculum
offset_warmup_fraction	0.5
conditioning_dropout_prob	0.2
seed_for_gen	42
optimizer 默认	AdamW, betas (0.9, 0.999), weight decay 1e-2, eps 1e-8

注意：训练脚本里 num_train_epochs 默认是 100，但因为启动脚本显式传了 --max_train_steps ，所以实际训练以该传入的steps 为准，epoch 数会被代码反推。


# evaluation

固定取val里的第一个场景的吧，使用 Evaluation-scripts/run_dist4d_shift1m2m4m_inference.sh 设置interval 2跑1m/2m/4m，然后再用同目录下的metrics脚本去做评估。
在val上随机抽10个样本做所有4个offsets的效果验证，每个offset不要单独保存而是一个样本一个6x6的拼接大视频：包括 “上方2行包括 target RGB 2x3全视角视频和 target Depth 2x3 全视角视频形成2x6、中间是inference input RGB + Depth 形成的2x6、下方是predict RGB + Depth 形成的2x6” 的拼接结果。最后计算predict RGB与target RGB之间的逐帧 SSIM/PSNR 均值和 mask-only的补全区域的 SSIM/PSNR 均值，depth之间就计算逐帧 depth RMSE、depth AbsRel 均值和 mask-only 的补全区域的逐帧 depth RMSE/AbsRel 均值。


