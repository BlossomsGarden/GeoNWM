#!/usr/bin/env python
"""Train a GLD-style MAE RGB decoder on frozen DVGT-1 features."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import AutoConfig


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def add_paths(repo_root: Path, dvgt_code_dir: str) -> None:
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(Path(dvgt_code_dir)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", default="explorations/dvgt_rgb_decoder_in_gld_style/results")
    parser.add_argument("--ckpt", default=None, help="Full resumable checkpoint.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--global-seed", type=int, default=None)
    return parser.parse_args()


def create_logger(logging_dir: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def import_any(candidates: list[str]):
    errors = []
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError("Could not import DVGT modules:\n" + "\n".join(errors))


def build_dvgt_model(ckpt_dir: Path, device: torch.device):
    module = import_any(["dvgt.models.architectures.dvgt1"])
    builder = getattr(module, "DVGT1")
    orig_hub_load = torch.hub.load

    def hub_load_no_pretrain(repo_or_dir, model, *args, **kwargs):
        if "dinov3" in str(repo_or_dir) and "pretrained" not in kwargs:
            kwargs["pretrained"] = False
        return orig_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = hub_load_no_pretrain
    old_cwd = os.getcwd()
    try:
        os.chdir(str(Path(ckpt_dir).parents[1] / "DVGT-code")) if False else None
        try:
            model = builder(dino_v3_weight_path=None, enable_ego_pose=False, enable_point=False)
        except Exception:
            model = builder(enable_ego_pose=False, enable_point=False)
    finally:
        torch.hub.load = orig_hub_load
        os.chdir(old_cwd)

    ckpt_files = sorted(list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.pth")) + list(ckpt_dir.glob("*.bin")))
    if not ckpt_files:
        raise FileNotFoundError(f"No DVGT checkpoint found under {ckpt_dir}")
    state = torch.load(ckpt_files[0], map_location="cpu")
    if isinstance(state, dict):
        state = state.get("model", state.get("state_dict", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded DVGT checkpoint {ckpt_files[0]} missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def extract_dvgt_features(
    model,
    images_01: torch.Tensor,
    feature_indices: list[int],
    per_level_layernorm: bool,
) -> torch.Tensor:
    """Return concatenated DVGT tokens as (B*V, N, C_total)."""
    if images_01.ndim != 5:
        raise ValueError(f"Expected images shaped (B,V,3,H,W), got {tuple(images_01.shape)}")
    images = images_01.unsqueeze(2)  # B,V,Tiny-view=1,3,H,W for DVGT aggregator
    out_list, patch_start_idx = model.aggregator(images)
    feats = []
    b, v, one = images.shape[:3]
    for idx in feature_indices:
        out = out_list[int(idx)]
        if out.ndim == 5:
            feat = out[:, :, :, patch_start_idx:, :].mean(dim=2)
        else:
            feat = out[:, patch_start_idx:, :]
            feat = feat.reshape(b, v, one, feat.shape[-2], feat.shape[-1]).mean(dim=2)
        if per_level_layernorm:
            feat = F.layer_norm(feat, (feat.shape[-1],))
        feats.append(feat)
    z = torch.cat(feats, dim=-1)
    return z.reshape(b * v, z.shape[-2], z.shape[-1]).contiguous()


def prepare_dataloader(dataset, batch_size, workers, test=False):
    from cut3r_data import get_data_loader

    return get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test,
        fixed_length=True,
        world_size=1,
        rank=0,
    )


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def calculate_adaptive_weight(recon_loss, gan_loss, layer, max_d_weight=1e4):
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    return torch.clamp(d_weight, 0.0, max_d_weight).detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    from disc import hinge_d_loss, vanilla_d_loss, vanilla_g_loss

    d_fn = {"hinge": hinge_d_loss, "vanilla": vanilla_d_loss}[disc_kind]
    g_fn = {"vanilla": vanilla_g_loss}[gen_kind]
    return d_fn, g_fn


def compute_psnr(gt, pred):
    mse = (gt.clamp(0, 1) - pred.clamp(0, 1)).pow(2).flatten(1).mean(dim=1)
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def save_visualization(gt_5d, pred_5d, save_path: Path, max_samples: int = 1):
    import numpy as np
    from PIL import Image

    rows = []
    for b in range(min(gt_5d.shape[0], max_samples)):
        gt = gt_5d[b].detach().cpu().permute(0, 2, 3, 1).numpy()
        pred = pred_5d[b].detach().cpu().permute(0, 2, 3, 1).numpy()
        rows.append(np.concatenate([gt[i] for i in range(gt.shape[0])], axis=1))
        rows.append(np.concatenate([pred[i] for i in range(pred.shape[0])], axis=1))
    vis = np.concatenate(rows, axis=0)
    Image.fromarray((vis.clip(0, 1) * 255).astype("uint8")).save(save_path)


def append_csv(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_loss_plot(csv_path: Path, png_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not plot losses: {exc}")
        return
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    steps = [int(r["step"]) for r in rows]
    plt.figure(figsize=(10, 5))
    for key in ["loss/recon", "loss/lpips", "loss/gan", "loss/disc", "loss/total"]:
        vals = [float(r[key]) for r in rows if key in r and r[key] != ""]
        if len(vals) == len(steps):
            plt.plot(steps, vals, label=key)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def save_resume_checkpoint(path, step, epoch, decoder, ema_decoder, optimizer, scheduler, disc, disc_optimizer, disc_scheduler):
    state = {
        "step": step,
        "epoch": epoch,
        "decoder": decoder.state_dict(),
        "ema_decoder": ema_decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def save_inference_checkpoint(path, step, epoch, ema_decoder, config):
    state = {
        "step": step,
        "epoch": epoch,
        "model_type": "ema_decoder",
        "format": "dvgt_stage1_mae_inference_only",
        "ema_decoder": ema_decoder.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_resume_checkpoint(path, decoder, ema_decoder, optimizer, scheduler, disc, disc_optimizer, disc_scheduler):
    ckpt = torch.load(path, map_location="cpu")
    if "decoder" not in ckpt and "ema_decoder" in ckpt:
        decoder.load_state_dict(ckpt["ema_decoder"])
        ema_decoder.load_state_dict(ckpt["ema_decoder"])
        step = int(ckpt.get("step", 0))
        if scheduler is not None and step > 0:
            scheduler.step(step)
        return int(ckpt.get("epoch", 0)), step

    decoder.load_state_dict(ckpt["decoder"])
    ema_decoder.load_state_dict(ckpt["ema_decoder"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        elif int(ckpt.get("step", 0)) > 0:
            scheduler.step(int(ckpt.get("step", 0)))
    if "disc" in ckpt:
        disc.load_state_dict(ckpt["disc"])
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        if disc_scheduler is not None and ckpt.get("disc_scheduler") is not None:
            disc_scheduler.load_state_dict(ckpt["disc_scheduler"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0))


@torch.no_grad()
def run_validation(
    dvgt,
    decoder,
    val_loader,
    device,
    cfg,
    lpips_fn,
    out_dir: Path,
    step: int,
    precision: str,
):
    from utils.metrics import compute_ssim

    decoder.eval()
    encoder_mean = IMAGENET_MEAN.to(device)
    encoder_std = IMAGENET_STD.to(device)
    feature_indices = [int(x) for x in cfg.dvgt.feature_indices]
    per_level_layernorm = bool(cfg.dvgt.get("per_level_layernorm", True))
    val_batches = int(cfg.reporting.get("val_batches", 200))
    vis_samples = int(cfg.reporting.get("vis_samples", 4))
    patch_size = int(cfg.decoder.patch_size)

    if hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "set_epoch"):
        val_loader.dataset.set_epoch(0)
    if hasattr(val_loader, "batch_sampler") and hasattr(val_loader.batch_sampler, "set_epoch"):
        val_loader.batch_sampler.set_epoch(0)

    use_autocast = precision in ("fp16", "bf16") and device.type == "cuda"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    sums = defaultdict(float)
    count = 0
    saved_vis = False

    for i, image_dict in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
        if i >= val_batches:
            break
        images_enc = torch.stack([d["img"] for d in image_dict], dim=1).to(device, non_blocking=True)
        b, v, c, h, w = images_enc.shape
        images_01 = images_enc * encoder_std[None] + encoder_mean[None]

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
            z = extract_dvgt_features(dvgt, images_01, feature_indices, per_level_layernorm)
            logits = decoder(z, input_size=(h, w), drop_cls_token=False).logits
            recon = decoder.unpatchify(logits, (h, w))
            recon_01 = recon * encoder_std.squeeze(0) + encoder_mean.squeeze(0)
        pred = recon_01.float().clamp(0, 1)
        gt = images_01.reshape(b * v, c, h, w).float().clamp(0, 1)

        n = gt.shape[0]
        sums["l1"] += F.l1_loss(pred, gt, reduction="none").mean(dim=(1, 2, 3)).sum().item()
        sums["psnr"] += compute_psnr(gt, pred).sum().item()
        sums["ssim"] += compute_ssim(gt.float(), pred.float()).sum().item()
        sums["lpips"] += lpips_fn(gt.float() * 2.0 - 1.0, pred.float() * 2.0 - 1.0).item() * n
        count += n

        if not saved_vis:
            vis_dir = out_dir / "vis"
            vis_dir.mkdir(parents=True, exist_ok=True)
            save_visualization(gt.view(b, v, c, h, w), pred.view(b, v, c, h, w), vis_dir / f"step_{step:07d}.png", vis_samples)
            saved_vis = True

    metrics = {f"val/{k}": v / max(count, 1) for k, v in sums.items()}
    metrics["val/images"] = count
    metrics["step"] = step
    report_path = out_dir / "eval" / f"step_{step:07d}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(args.config)
    add_paths(repo_root, cfg.dvgt.code_dir)

    from disc import LPIPS, build_discriminator
    from stage1.decoders import GeneralDecoder_Variable
    from utils.optim_utils import build_optimizer, build_scheduler

    precision = args.precision or str(cfg.training.get("precision", "bf16"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
    seed = args.global_seed if args.global_seed is not None else int(cfg.training.get("global_seed", 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    experiment_index = len([p for p in results_dir.glob("*") if p.is_dir()])
    experiment_dir = results_dir / f"{experiment_index:03d}-RAE_DVGT_MAE-bf16"
    checkpoint_dir = experiment_dir / "checkpoints"
    resume_dir = experiment_dir / "checkpoints_resume"
    report_dir = experiment_dir / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger(str(experiment_dir))
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Config: {args.config}")

    train_loader = prepare_dataloader(cfg.train_dataset, int(cfg.training.batch_size), int(cfg.training.num_workers), test=False)
    val_loader = prepare_dataloader(cfg.test_dataset, int(cfg.training.batch_size), int(cfg.training.num_workers), test=True)
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches.")

    old_cwd = os.getcwd()
    os.chdir(str(cfg.dvgt.code_dir))
    try:
        dvgt = build_dvgt_model(Path(cfg.dvgt.ckpt_dir), device)
    finally:
        os.chdir(old_cwd)

    dec_config = AutoConfig.from_pretrained(str(cfg.decoder.config_path))
    dec_config.hidden_size = int(cfg.decoder.hidden_size)
    dec_config.patch_size = int(cfg.decoder.patch_size)
    base_image_size = tuple(int(x) for x in cfg.decoder.get("base_image_size", [448, 448]))
    decoder = GeneralDecoder_Variable(dec_config, base_image_size=base_image_size).to(device)
    ema_decoder = deepcopy(decoder).to(device).eval()
    ema_decoder.requires_grad_(False)

    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    optimizer, _ = build_optimizer([p for p in decoder.parameters() if p.requires_grad], training_cfg)
    scheduler = build_scheduler(optimizer, steps_per_epoch, training_cfg)[0] if training_cfg.get("scheduler") else None

    gan_cfg = OmegaConf.to_container(cfg.gan, resolve=True)
    disc_cfg = gan_cfg["disc"]
    loss_cfg = gan_cfg["loss"]
    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    disc_optimizer, _ = build_optimizer([p for p in discriminator.parameters() if p.requires_grad], disc_cfg)
    disc_scheduler = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)[0] if disc_cfg.get("scheduler") else None
    disc_loss_fn, gen_loss_fn = select_gan_losses(loss_cfg.get("disc_loss", "hinge"), loss_cfg.get("gen_loss", "vanilla"))

    lpips = LPIPS().to(device).eval()
    for p in lpips.parameters():
        p.requires_grad_(False)

    scaler = GradScaler() if precision == "fp16" else None
    autocast_kwargs = dict(enabled=precision in ("fp16", "bf16"), dtype=torch.float16 if precision == "fp16" else torch.bfloat16)

    start_epoch = 0
    global_step = 0
    if args.ckpt:
        start_epoch, global_step = load_resume_checkpoint(
            Path(args.ckpt), decoder, ema_decoder, optimizer, scheduler,
            discriminator, disc_optimizer, disc_scheduler,
        )
        logger.info(f"Resumed from {args.ckpt} epoch={start_epoch} step={global_step}")

    encoder_mean = IMAGENET_MEAN.to(device)
    encoder_std = IMAGENET_STD.to(device)
    feature_indices = [int(x) for x in cfg.dvgt.feature_indices]
    per_level_layernorm = bool(cfg.dvgt.get("per_level_layernorm", True))
    patch_size = int(cfg.decoder.patch_size)
    recon_weight = float(loss_cfg.get("recon_weight", 1.0))
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 1.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_step = int(loss_cfg.get("disc_start", 0)) * steps_per_epoch
    disc_update_step = int(loss_cfg.get("disc_upd_start", loss_cfg.get("disc_start", 0))) * steps_per_epoch
    lpips_start_step = int(loss_cfg.get("lpips_start", 0)) * steps_per_epoch
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    ema_decay = float(cfg.training.ema_decay)
    log_interval = int(cfg.training.log_interval)
    val_interval = int(cfg.training.get("validation_interval", 10000))
    infer_interval = int(cfg.training.get("inference_checkpoint_interval", 10000))
    last_layer = decoder.decoder_pred.weight
    metrics_csv = report_dir / "metrics.csv"

    logger.info(f"DVGT feature_indices={feature_indices}, per_level_layernorm={per_level_layernorm}")
    logger.info(f"Training bs={cfg.training.batch_size}, steps_per_epoch={steps_per_epoch}, patch_size={patch_size}")
    logger.info(f"GAN starts at step {gan_start_step}, disc update starts at step {disc_update_step}")

    for epoch in range(start_epoch, int(cfg.training.epochs)):
        decoder.train()
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        epoch_metrics = defaultdict(lambda: torch.zeros((), device=device))
        num_batches = 0
        pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch}/{cfg.training.epochs}")
        for image_dict in pbar:
            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            train_disc = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0

            images_enc = torch.cat([d["img"].unsqueeze(1) for d in image_dict], dim=1).to(device, non_blocking=True)
            b, v, c, h, w = images_enc.shape
            if h % patch_size != 0 or w % patch_size != 0:
                raise ValueError(f"Image size {(h, w)} must be divisible by DVGT patch_size={patch_size}")
            images_01 = images_enc * encoder_std[None] + encoder_mean[None]
            images_flat = images_01.reshape(b * v, c, h, w).contiguous()
            real_normed = images_flat * 2.0 - 1.0

            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()
            with autocast(**autocast_kwargs):
                with torch.no_grad():
                    z = extract_dvgt_features(dvgt, images_01, feature_indices, per_level_layernorm)
                logits = decoder(z, input_size=(h, w), drop_cls_token=False).logits
                recon = decoder.unpatchify(logits, (h, w))
                recon_01 = recon * encoder_std.squeeze(0) + encoder_mean.squeeze(0)
                recon_pm1 = recon_01 * 2.0 - 1.0
                rec_loss = F.l1_loss(recon_01, images_flat)
                lpips_loss = lpips(real_normed, recon_pm1) if use_lpips else rec_loss.new_zeros(())
                recon_total = recon_weight * rec_loss + perceptual_weight * lpips_loss
                if use_gan:
                    i_crop, j_crop, h_crop, w_crop = T.RandomCrop.get_params(recon_pm1, output_size=(224, 224))
                    fake_aug = disc_aug.aug(TF.crop(recon_pm1, i_crop, j_crop, h_crop, w_crop))
                    logits_fake, _ = discriminator(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(recon_total)

            if use_gan:
                adaptive_weight = calculate_adaptive_weight(recon_total, gan_loss, last_layer, max_d_weight)
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = recon_total

            if scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            update_ema(ema_decoder, decoder, ema_decay)

            disc_metrics = {}
            if train_disc:
                discriminator.train()
                decoder.eval()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast(**autocast_kwargs):
                        with torch.no_grad():
                            logits_d = decoder(z, input_size=(h, w), drop_cls_token=False).logits
                            recon_d = decoder.unpatchify(logits_d, (h, w))
                            recon_d_01 = recon_d * encoder_std.squeeze(0) + encoder_mean.squeeze(0)
                            fake_det = (recon_d_01 * 2.0 - 1.0).clamp(-1.0, 1.0)
                            fake_det = torch.round((fake_det + 1.0) * 127.5) / 127.5 - 1.0
                        i_crop, j_crop, h_crop, w_crop = T.RandomCrop.get_params(real_normed, output_size=(224, 224))
                        fake_input = disc_aug.aug(TF.crop(fake_det, i_crop, j_crop, h_crop, w_crop))
                        real_input = disc_aug.aug(TF.crop(real_normed, i_crop, j_crop, h_crop, w_crop))
                        logits_fake_d, logits_real_d = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real_d, logits_fake_d)
                        accuracy = (logits_real_d > logits_fake_d).float().mean()
                    d_loss.backward()
                    disc_optimizer.step()
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                    disc_metrics = {
                        "loss/disc": d_loss.detach(),
                        "disc/accuracy": accuracy.detach(),
                    }
                discriminator.eval()
                decoder.train()

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0:
                stats = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "loss/disc": disc_metrics.get("loss/disc", torch.zeros(())).item() if disc_metrics else 0.0,
                    "gan/weight": adaptive_weight.detach().item(),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                logger.info(", ".join(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()))
                append_csv(metrics_csv, stats)

            global_step += 1

            if infer_interval > 0 and global_step % infer_interval == 0:
                ckpt_path = checkpoint_dir / f"{global_step:07d}.pt"
                save_inference_checkpoint(ckpt_path, global_step, epoch, ema_decoder, cfg)
                val_stats = run_validation(dvgt, ema_decoder, val_loader, device, cfg, lpips, report_dir, global_step, precision)
                logger.info("[Val] " + ", ".join(f"{k}: {v:.6f}" for k, v in val_stats.items() if isinstance(v, float)))
                save_loss_plot(metrics_csv, report_dir / "loss_curve.png")

            elif val_interval > 0 and global_step % val_interval == 0:
                val_stats = run_validation(dvgt, ema_decoder, val_loader, device, cfg, lpips, report_dir, global_step, precision)
                logger.info("[Val] " + ", ".join(f"{k}: {v:.6f}" for k, v in val_stats.items() if isinstance(v, float)))
                save_loss_plot(metrics_csv, report_dir / "loss_curve.png")

        if num_batches > 0:
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"epoch/{k}: {(epoch_metrics[k] / num_batches).item():.6f}" for k in ["recon", "lpips", "gan", "total"])
            )
            save_resume_checkpoint(
                resume_dir / f"epoch_{epoch + 1:04d}.pt",
                global_step,
                epoch,
                decoder,
                ema_decoder,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )


if __name__ == "__main__":
    main()
