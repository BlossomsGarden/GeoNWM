# Reproduce GLD DA3 RGB Decoder On Waymo

## 0) Quick Reuse (336x224, verified run)

- Keep these files:
  - `reproduce_gld_on_waymo/DA3_stage1_mae_waymo_336x224.yaml`
  - `reproduce_gld_on_waymo/run_train_stage1_mae_waymo.sh`
  - `reproduce_gld_on_waymo/check_waymo_stage1_dataset.py`
  - `reproduce_gld_on_waymo/decoder_ViTXL/config.json`
- Waymo processed root:
  - `/data/wlh/FreeDrive/data/waymo/processed-test`
- Trainer entry used by launch script:
  - `src/train_stage1_mae.py`
- Default output directory:
  - `reproduce_gld_on_waymo/results/stage1-mae-waymo-336x224`
- Path caveat:
  - Current yaml uses `reprojuce_gld_on_waymo/...` in `decoder.config_path`.
    If your folder is `reproduce_gld_on_waymo`, update that path accordingly.

Start commands (run from repo root):

```bash
python reproduce_gld_on_waymo/check_waymo_stage1_dataset.py --split train --config reproduce_gld_on_waymo/DA3_stage1_mae_waymo_336x224.yaml
python reproduce_gld_on_waymo/check_waymo_stage1_dataset.py --split test --config reproduce_gld_on_waymo/DA3_stage1_mae_waymo_336x224.yaml
bash reproduce_gld_on_waymo/run_train_stage1_mae_waymo.sh 1
```

This folder reuses GLD stage-1 MAE RGB decoder training on Waymo:

```bash
/data/wlh/FreeDrive/data/waymo/processed-test
```

## 1) Folder/file meanings

- `DA3_stage1_mae_waymo_*.yaml`: training/eval configs at different resolutions.
- `check_waymo_stage1_dataset.py`: sanity-checks dataset split and tensor shape.
- `run_train_stage1_mae_waymo.sh`: launch stage-1 MAE training (torchrun).
- `eval_stage1_mae_waymo.py`: quantitative eval script (L1/PSNR/SSIM/LPIPS).
- `run_eval_stage1_mae_waymo.sh`: one-line wrapper to run evaluation.
- `decoder_ViTXL/`: local decoder config passed to `AutoConfig`.
- `results/`: training outputs (created during training).
- `reports/`: saved GT-vs-pred visualization images from eval.

Typical training output structure:

```text
reproduce_gld_on_waymo/results/stage1-mae-waymo-336x224/
  └── 000-RAE_DA3_MAE-bf16/
      ├── log.txt
      ├── checkpoints_resume/
      │   ├── epoch_0001.pt
      │   ├── epoch_0002.pt
      │   └── ...
      └── checkpoints/
          ├── 0010000.pt
          ├── 0020000.pt
          └── ...
```

## 2) Dataset split policy

- `segment-*` scenes -> train split
- non-`segment-*` scenes -> test split
- image path format: `scene/images/{frame}_{camera}.jpg`
- default cameras: `[0, 1, 2]`

## 3) Resolution/aspect policy

- Keep ratio scaling from original image (example: `1920x1280 -> 672x448` or `336x224`).
- `strict_aspect=True`: if aspect mismatch exists, dataloader raises error.

## 4) Training commands

Sanity check first:

```bash
python reproduce_gld_on_waymo/check_waymo_stage1_dataset.py --split train --config reproduce_gld_on_waymo/DA3_stage1_mae_waymo_336x224.yaml
python reproduce_gld_on_waymo/check_waymo_stage1_dataset.py --split test --config reproduce_gld_on_waymo/DA3_stage1_mae_waymo_336x224.yaml
```

Start training:

```bash
bash reproduce_gld_on_waymo/run_train_stage1_mae_waymo.sh 1
```

Resume training:

```bash
bash reproduce_gld_on_waymo/run_train_stage1_mae_waymo.sh 1 \
  reproduce_gld_on_waymo/results/stage1-mae-waymo-336x224/000-RAE_DA3_MAE-bf16/checkpoints_resume/epoch_0001.pt
```

## 5) Quantitative metrics (PSNR / SSIM / LPIPS)

Evaluate one checkpoint on test split:

```bash
CUDA_VISIBLE_DEVICES=0 bash reproduce_gld_on_waymo/run_eval_stage1_mae_waymo.sh \
  reproduce_gld_on_waymo/results/stage1-mae-waymo-336x224/000-RAE_DA3_MAE-bf16/checkpoints/0010000.pt
```

Equivalent raw command:

```bash
CUDA_VISIBLE_DEVICES=0 python reproduce_gld_on_waymo/eval_stage1_mae_waymo.py \
  --config reproduce_gld_on_waymo/DA3_stage1_mae_waymo_336x224.yaml \
  --ckpt reproduce_gld_on_waymo/results/stage1-mae-waymo-336x224/000-RAE_DA3_MAE-bf16/checkpoints/0010000.pt \
  --split test \
  --max-batches 200 \
  --use-ema \
  --precision bf16 \
  --save-vis-dir reproduce_gld_on_waymo/reports/step10000_test \
  --save-vis-every 50
```

Metrics are computed in RGB space `[0,1]`:

- `L1`: mean absolute reconstruction error
- `PSNR`: higher is better
- `SSIM`: higher is better
- `LPIPS`: lower is better

The script prints JSON and also writes an output file such as:

```text
reproduce_gld_on_waymo/results/stage1-mae-waymo-336x224/000-RAE_DA3_MAE-bf16/eval_test_step5000.json
reproduce_gld_on_waymo/results/stage1-mae-waymo-336x224/000-RAE_DA3_MAE-bf16/eval_test_step10000.json
```

Checkpoint policy:

- `checkpoints_resume/epoch_xxxx.pt`: full resumable snapshots (large).
- `checkpoints/00xxxxx.pt`: inference-only checkpoints saved every 10000 steps (small).
- `eval_stage1_mae_waymo.py` supports both formats.

## 6) Selecting best checkpoint

Evaluate multiple checkpoints and compare `LPIPS` (lower) + `PSNR/SSIM` (higher).
For your patch artifact issue, LPIPS and saved visualizations in `reports/` are usually the quickest signal.

## 7) What stays identical to GLD

`src/train_stage1_mae.py` still keeps GLD training logic/tricks:

- EMA decoder
- LPIPS loss
- DINO discriminator + adaptive GAN weighting
- cosine LR schedule
- bf16 mixed precision
