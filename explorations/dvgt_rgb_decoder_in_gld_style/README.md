# GLD-Style DVGT RGB Decoder On Waymo

## 0) Quick Reuse (672x448, verified run)

- Keep these files:
  - `explorations/dvgt_rgb_decoder_in_gld_style/configs/DVGT_stage1_mae_waymo_672x448.yaml`
  - `explorations/dvgt_rgb_decoder_in_gld_style/scripts/train_stage1_mae_dvgt.py`
  - `explorations/dvgt_rgb_decoder_in_gld_style/scripts/run_train_stage1_mae_dvgt.sh`
  - `explorations/dvgt_rgb_decoder_in_gld_style/decoder_ViTXL/config.json`
- Waymo processed root:
  - `/data/wlh/FreeDrive/data/waymo/processed-test`
- DVGT dependencies:
  - `dvgt.code_dir: /data/wlh/GLD/DVGT-code`
  - `dvgt.ckpt_dir: /data/wlh/GLD/DVGT-model/DVGT-1`
- Default output directory:
  - `explorations/dvgt_rgb_decoder_in_gld_style/results/stage1-mae-dvgt-waymo-672x448`

Start commands (run from repo root):

```bash
export CUDA_VISIBLE_DEVICES=3
bash explorations/dvgt_rgb_decoder_in_gld_style/scripts/run_train_stage1_mae_dvgt.sh \
  explorations/dvgt_rgb_decoder_in_gld_style/configs/DVGT_stage1_mae_waymo_672x448.yaml
```

This experiment mirrors `reproduce_gld_on_waymo` while replacing DA3 feature
extraction with frozen DVGT-1 aggregator features.

Remote path:

```text
/data/wlh/GLD/code/explorations/dvgt_rgb_decoder_in_gld_style
```

Key choices:

- Dataset: `/data/wlh/FreeDrive/data/waymo/processed-test`
- Train split: `segment-*`
- Test split: non-`segment-*` via `WaymoStage1_Multi`
- Resolution: `672x448`
- GPU: physical `cuda:3` (`CUDA_VISIBLE_DEVICES=3`)
- DVGT feature layers: `[4, 11, 17, 23]`
- DVGT patch size: `16`
- Decoder: GLD `GeneralDecoder_Variable`
- Loss/tricks: L1 + LPIPS, EMA, DINO discriminator/adaptive GAN, cosine LR
- Inference checkpoints/validation/visualization: every `10000` steps

Run on remote:

```bash
cd /data/wlh/GLD/code
bash explorations/dvgt_rgb_decoder_in_gld_style/scripts/start_remote_training.sh
```

Watch:

```bash
tail -f /data/wlh/GLD/code/explorations/dvgt_rgb_decoder_in_gld_style/train_672x448.log
```

Outputs:

```text
explorations/dvgt_rgb_decoder_in_gld_style/results/stage1-mae-dvgt-waymo-672x448/
  000-RAE_DVGT_MAE-bf16/
    checkpoints/          # inference-only EMA weights every 10000 steps
    checkpoints_resume/   # full resumable snapshots every epoch
    reports/
      metrics.csv
      loss_curve.png
      eval/step_*.json
      vis/step_*.png
    log.txt
```

Evaluate a saved checkpoint:

```bash
CUDA_VISIBLE_DEVICES=3 python explorations/dvgt_rgb_decoder_in_gld_style/scripts/eval_stage1_mae_dvgt.py \
  --config explorations/dvgt_rgb_decoder_in_gld_style/configs/DVGT_stage1_mae_waymo_672x448.yaml \
  --ckpt explorations/dvgt_rgb_decoder_in_gld_style/results/stage1-mae-dvgt-waymo-672x448/000-RAE_DVGT_MAE-bf16/checkpoints/0010000.pt \
  --precision bf16
```
