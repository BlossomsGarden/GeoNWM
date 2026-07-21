# 本文档目标

在nuscenes数据集，使用 DepthAnythingV3 为几何基础模型（GFM），实现以 compressed DA3 Block1 feature map为条件的新视角下的 compressed DA3 Block1 feature map 生成。生成完毕后使用DA3_Adapter将其恢复为原DA3 Block1 feature map，再经过前向传播、DPT-Head解码获得新视角下的深度等等多模态下游输出。



# 方案A

## DA3_Adapter

### 数据集构造
使用 REMOTE_SERVER.md 中的远程服务器上的 /data2/DATA/AutoDrive/nuscenes/v1.0-trainval 路径中的nuscenes数据集以及给出的samples关键帧的记录文件。取图片时要按照resize、居中crop的操作来进行分辨率转换，是那种能够和相机内参相对应的resize/crop方式。训练脚本随机采样某个场景中的一段9帧序列，转为1456x840分辨率输入DA3进行前向传播并解码获取深度，同时记录Block1的raw feature map作为监督目标；同时再把这个序列转为 832x480 分辨率给 Wan VAE 编码作为 KL 对齐目标。

### 训练
基于 DA3-METRIC-LARGE 模型，其DINO一共24个block，取出第1个block（该模型cat_token = false）的raw feature map，一共1024维通道，将它压缩进入16维通道。训练loss几乎沿用Gen3R，只是不包括与camera有关的项，具体来说包括：recon_loss、similarity_loss、align_loss三者加权。其中recon_loss包括image_token_loss（smooth L1）、depth_loss和point_loss（重建后继续执行前向传播获得Dual-DPT解码所需的层级后解码出结果来再计算loss），similarity_loss为image_cosine_loss，align_loss复用。


## Geometric_Latent_DiT

### 数据集构造
训练脚本随机采样某个场景中的一段9帧序列。。。

### 训练
使用 Wan2.1-T2V 模型，其权重全量复制保留，但是将该“832x480分辨率的RGB视频生成模型”改为“DA3 Block1 Feature生成模型”，基于warped DA3 Block1 Feature序列（9帧）为条件，生成完整的9帧 DA3 Block1 Feature。具体实现方式为在noisy latent的16通道再外接16个通道，它是 1456x840x9帧 的原视频经过 DA3 编码后形成 104x60x1024x9帧 的 DA3 Block1 Feature map，经过 DA3_Adapter 形成 104x60x16x9帧 的 compressed DA3 Block1 feature map。然后前面提到的原视频经过 DA3 前向传播解码出深度图后下采样到 104x60分辨率，再逐帧对 compressed DA3 Block1 feature map 进行 warp 获得新视角下的 warped compressed DA3 Block1 feature map 104x60x16x9帧。将这个warped compressed DA3 Block1 feature map按通道拼接到noisy latent上执行有条件训练，最终监督生成出完整的真实新视角下的 compressed DA3 Block1 feature map。


# 方案B

## DA3_Adapter
方案A使用的DA3-METRIC-LARGE 只有metric深度而无法估计位姿，即没有camera token，我担心训练不出效果来。
这里将基模型替换为 DA3-GIANT-1.1，再把cam也加入训练监督中

### 数据集构造
大体上同方案A，只是增加一项：记录相机同步随1456x840分辨率变化的内参变化

### 训练
大体同方案A，只是DINO一共40个block了。loss方面的变化，recon_loss包括image_token_loss和camera_token_loss（smooth L1）、camera_loss和depth_loss和point_loss（重建后继续执行前向传播获得Dual-DPT解码所需的层级后解码出结果来再计算loss），similarity_loss包括camera_cosine_loss和image_cosine_loss，align_loss复用。


## Geometric_Latent_DiT

同方案A。



# 方案C

## DA3_Adapter
大体上等同方案B，只不过在loss的recon_loss部分除了depth_loss和point_loss再增加一项rgb_loss，即重建后继续执行前向传播获得Dual-DPT解码所需的层级后解码出像素级结果来再计算loss。


## Geometric_Latent_DiT

同方案A。




# 方案D

## DA3_Adapter
数据集构造与训练同方案A

## Geometric_Latent_DiT
方案A是只生成新视角下的 compressed DA3 Block1 feature map，我担心很难训练，可能有RGB辅助的情况下会更好？相当于充分利用已有的RGB生成先验，然后让 compressed DA3 Block1 feature map 生成的能力能够更加容易地训练出来。
具体实施上，使用类似Gen3R的通道拼接思想，将类似于方案A里获得的新视角下的 warped compressed DA3 Block1 feature map 104x60x16x9帧和原832x480分辨率的RGB视频经过Wan VAE编码出的104x60x16x9帧在width上拼接形成208x60x16x9帧，并且将这个维度作为新的noisy latent(x)的维度。具体control latent(y)的构建则是104x60x16x9帧的全0在width上拼接原未经过投影的 compressed DA3 Block1 feature map 形成208x60x16x9帧，在送入Transformer前将x和y按照通道拼接形成32通道。







# 验收期望
1、公用数据集加载、工具函数等逻辑可以直接落在WarpGFM文件夹下创建子文件夹归类，在具体启动训练前优先返回给我一个构造出的数据集监督对
2、将方案中DA3_Adapter有关逻辑写入WarpGFM/da3_adapter文件夹下
3、将方案中Geometric_Latent_DiT有关逻辑写入WarpGFM/geo_latent_dit文件夹下
4、方案中DA3_Adapter和Geometric_Latent_DiT的训推启动代码直接在 WarpGFM 文件夹下分别落四个文件
5、整个文件夹架构应当清晰易读，且落README文件介绍架构与启动方式



# TBD
1、Text Prompt 该怎么写？还是空着？
2、Gen3R这种width拼接是完全避免了RGB和Geo生成互相影响吗？如果完全隔离那么方案B就失效了
3、Gen3R对每个 Wan/DiT 训练样本：随机选一个 prompt 行，并做 prompt 清洗。20% 概率把 text prompt 置空，用于 CFG。都有什么含义？而且它resize + center crop 到 560 x 560，归一化 [0,1]。这个是否能够借鉴？


# 背景补充

## DA3系列模型

DA3-Base                                DINOv2 ViT-B/14                  [5,7,9,11]       768*2
DA3-NESTED-GIANT-LARGE-1.1(Metric)      DINOv2 ViT-L/14                  [4,11,17,23]     1024
DA3-NESTED-GIANT-LARGE-1.1(AnyView)     DINOv2 ViT-G/14                  [19,27,33,39]    1536*2
DA3-GIANT-1.1                           DINOv2 ViT-G/14                  [19,27,33,39]    1536*2
DA3-METRIC-LARGE                        DINOv2 ViT-L/14                  [4,11,17,23]     1024

DA3 论文说：如果有相机参数，就把每个 view 的相机参数编码成一个 camera token，在 DINO 第 alt_start 层替换 token 0
之后参与 local/global attention；如果没有，就用 shared learnable token，而且是第一个 view 用 ref_token，其余 view 用 src_token。Dual-DPT 只吃 patch tokens；camera token 会通过 attention 影响 patch tokens，但它本身被单独拿出来给 CameraDec 回归 pose。

*2 的来源是 DA3 代码里的 cat_token=true：取某层输出时，把 local attention feature 和 global attention feature 在 channel 维 concat。Metric 分支 cat_token=false，所以不翻倍。global attn 的输出确实已经基于前一个 local attn 的结果，所以从信息流角度它不是“完全缺失 local 信息”。但 DA3 AnyView 仍然选择 cat，主要是为了把两种状态显式都交给 DPT——[per-view/local feature ; cross-view/global feature]。Metric 分支不 cat，不是因为它觉得 global 已经包含 local 就够了，而是因为 Metric 分支根本不走这套 local/global alternating cross-view 机制。这意味着它就是普通单图 DINO feature extraction，没有 camera token 注入，没有 cross-view global attention。每层只有一条普通 token 流。



## VGGT标准开源模型

VGGT-1B                                 DINOv2 ViT-L/14 with registers   [4,11,17,23]     1024*2




## GLD实现细节

使用 DA3-Base 模型，




## Gen3R实现细节

### camera token处理
使用 VGGT-1B 模型，由于 camera token 只有一个，但是每张图片、每层 Layer 的 aggregator 有 H'x W'个patch tokens（每个patch有2048通道维度），不能简单地在 H' x W' 上+1，而是采用了将 camera token broadcast到 H'x W'的方式，最终总维度为 H'x W' x (4xLayer + 1xbroadcastcamera)x2048 


### 条件注入
噪声目标x latent：[B,16,f,70,140]，其中最后一个维度的左半 70 是原生 appearance latent，右半 70 是 geometry latent。
控制条件y latent： [B,20,f,70,140]，其中它是由 control_latents_16 [B,16,f,70,140] 按channel拼接 mask [B,4,f,70,140] 形成。
最后进DiT时继续按channel把 x⊕y 拼接成 [B,36,f,70,140]

其中f=(F+3)//4，一个 latent timestep 对应 4 个原始帧位置。所以 mask 先是 [B,F,70,140]，再 reshape/transpose 成 [B,4,f,70,140]，这 4 个 channel 表示同一个 latent timestep 里 4 个原始帧槽位的 mask。单个 [B,1,f,70,140] 只能说“这个 latent timestep 有条件”，但分不清 4 个原始帧里到底哪个被条件控制。

训练时：噪声目标latent分别来自wan_vae(真实视频)、da3_adapter(DA3_Block1)，控制latent中的condition_latent来自于[wan_vae编码条件帧序列; geo_adapter编码条件帧的vggt feature]，mask只在条件帧的左侧RGB区置1，其他全是0，甚至条件帧的Geo区也是0。按照wan2.1-I2V的语义，mask区域应该是直接复制。
推理时：噪声目标latent就是随机初始化的噪声，控制latent中的condition_latent来自于[wan_vae编码条件帧序列; 全零]，mask与训练机制同。这里存在一个bug，训练时虽然传入
