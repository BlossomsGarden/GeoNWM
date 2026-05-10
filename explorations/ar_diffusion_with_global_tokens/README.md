# AR Diffusion With Global Tokens

## 0) Quick Reuse (v2_7h1f_336x224, verified run)

- Keep these files:
  - `explorations/ar_diffusion_with_global_tokens/configs/waymo_da3_ar_cdit_formal_v2_7h1f_336x224.yaml`
  - `explorations/ar_diffusion_with_global_tokens/scripts/train_ar_da3.py`
  - `explorations/ar_diffusion_with_global_tokens/scripts/launch_v2_7h1f_train.sh`
- Waymo processed root:
  - `/data/wlh/FreeDrive/data/waymo/processed-test`
- DA3 dependencies:
  - `da3.encoder_pretrained_path: /data/wlh/GLD/model/pretrained_models/da3`
  - `da3.decoder_config_path: reprojuce_gld_on_waymo/decoder_ViTXL`
  - `da3.rgb_decoder_ckpt: /data/wlh/GLD/code/reprojuce_gld_on_waymo/results/stage1-mae-waymo-336x224/001-RAE_DA3_MAE-bf16/checkpoints/1100000.pt`
- Path caveat:
  - This config keeps the historical folder name `reprojuce_gld_on_waymo`.
    If your folder is `reproduce_gld_on_waymo`, update both DA3 path fields.
- Default output directory:
  - `/data/wlh/GLD/outputs/ar_diffusion_with_global_tokens/waymo_da3_cdit_formal_v2_7h1f_336x224`

Start commands (run from repo root):

```bash
bash explorations/ar_diffusion_with_global_tokens/scripts/launch_v2_7h1f_train.sh
```

This exploration implements a low-resolution autonomous-driving world-model
prototype plus a formal cDiT-scale training configuration:

```text
6 history frames x 3 Waymo cameras
+ future ego trajectory / camera poses
-> metric camera Plucker ray maps in the last-history ego frame
-> joint denoising of structured DA3 latents [T, V, N, C]
+ recurrent global/world tokens
-> per-view DA3 MAE RGB decoder
```

The first target is code-path validation on the remote `cuda:3` 48GB GPU.  It
uses Waymo processed data under `/data/wlh/FreeDrive/data/waymo/processed-test`,
trains on `segment-*` folders, and reserves the non-`segment-*` folder for
validation.

Two denoisers are available:

- `factorized_global` in `waymo_da3_ar_336x224.yaml`: the small validation
  model.
- `cdit_formal` in `waymo_da3_ar_cdit_formal_336x224.yaml`: a 657M-parameter
  formal AR diffusion model with recurrent world tokens, target self-attention,
  cross-attention to global context, AdaLN timestep/action conditioning, and
  gradient checkpointing.

## Remote Quick Start

From the repo root on the remote server:

```bash
cd /data/wlh/GLD/code
source /data/wlh/miniconda3/etc/profile.d/conda.sh
conda activate gld
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/data/wlh/GLD/code/src:${PYTHONPATH}

python explorations/ar_diffusion_with_global_tokens/scripts/train_ar_da3.py \
  --config explorations/ar_diffusion_with_global_tokens/configs/waymo_da3_ar_336x224.yaml \
  --max-steps 20 \
  --val-every 10 \
  --ckpt-every 20
```

For the formal background run on `cuda:3`:

```bash
bash explorations/ar_diffusion_with_global_tokens/scripts/launch_formal_train.sh
tail -f /data/wlh/GLD/outputs/ar_diffusion_with_global_tokens/waymo_da3_cdit_formal_336x224/train_formal.log
```

For a 20-frame autoregressive rollout after training:

```bash
python explorations/ar_diffusion_with_global_tokens/scripts/train_ar_da3.py \
  --config explorations/ar_diffusion_with_global_tokens/configs/waymo_da3_ar_336x224.yaml \
  --mode rollout \
  --ckpt /data/wlh/GLD/outputs/ar_diffusion_with_global_tokens/waymo_da3_336x224/checkpoints/last.pt \
  --rollout-frames 20
```

## Design Notes

- The generated object is the native DA3 four-level feature tensor, not a video
  VAE latent.  The denoiser internally projects 6144-dim tokens to a smaller
  hidden size, but the supervised target and decoder input stay in DA3 feature
  space.
- Global tokens are a recurrent world-state memory.  They are initialized by
  cross-attending to history latents and updated after each autoregressive
  forward step by cross-attending to generated future latents.
- Joint generation means the denoiser processes `[T_future, V, N]` tokens in
  one structured tensor with factorized spatial, cross-view, temporal, and
  global-memory attention.  The RGB decoder is still per-view in this first
  prototype.
- Pose conditioning is dense: every future patch receives a metric Plucker ray
  built from `ego_pose`, camera extrinsic, camera intrinsic, and the chosen
  history-last ego reference frame.
