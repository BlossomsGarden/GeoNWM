#!/usr/bin/env python
"""AR diffusion models with recurrent global tokens in native DA3 space."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def add_repo_to_path() -> Path:
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "src"))
    return repo


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(args) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if args.local_rank is not None:
        local_rank = int(args.local_rank)
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA when WORLD_SIZE > 1.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available() and str(args.device or "").startswith("cuda"):
        torch.cuda.set_device(torch.device(str(args.device)).index or 0)
    return distributed, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def rank0_print(rank: int, *args, **kwargs) -> None:
    if is_main_process(rank):
        print(*args, **kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_mean(tensor: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return tensor
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


def load_txt_matrix(path: Path, shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    return arr.reshape(shape).astype(np.float32)


def yaw_from_matrix(mat: torch.Tensor) -> torch.Tensor:
    return torch.atan2(mat[..., 1, 0], mat[..., 0, 0])


def scale_intrinsics(values: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    # Waymo intrinsics are [fx, fy, cx, cy, ...].
    src_w, src_h = src_size
    dst_h, dst_w = dst_size
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out = values.astype(np.float32).copy()
    out[0] *= sx
    out[1] *= sy
    out[2] *= sx
    out[3] *= sy
    return out


def make_plucker_rays(
    c2ref: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    patch_size: int,
    pose_scale: float,
) -> torch.Tensor:
    """Return patch-level Plucker rays as (N, 6) in the reference ego frame."""
    device = c2ref.device
    dtype = c2ref.dtype
    h, w = image_size
    hp = h // patch_size
    wp = w // patch_size
    yy, xx = torch.meshgrid(
        torch.arange(hp, device=device, dtype=dtype),
        torch.arange(wp, device=device, dtype=dtype),
        indexing="ij",
    )
    u = (xx + 0.5) * patch_size
    v = (yy + 0.5) * patch_size
    fx, fy, cx, cy = intrinsics[:4]
    dirs_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    dirs_cam = F.normalize(dirs_cam.reshape(-1, 3), dim=-1)
    rot = c2ref[:3, :3]
    origin = c2ref[:3, 3] / float(pose_scale)
    dirs_ref = F.normalize(dirs_cam @ rot.T, dim=-1)
    origins = origin.view(1, 3).expand_as(dirs_ref)
    moment = torch.cross(origins, dirs_ref, dim=-1)
    return torch.cat([dirs_ref, moment], dim=-1)


class WaymoRigWindowDataset(Dataset):
    """Aligned multi-camera temporal windows from the processed Waymo layout."""

    def __init__(
        self,
        root: str,
        split: str,
        camera_ids: list[int],
        image_size: tuple[int, int],
        history_frames: int,
        future_frames: int,
        frame_stride: int,
        window_stride: int,
        pose_scale: float,
        max_scenes: int = 0,
        patch_size: int = 14,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.camera_ids = [str(c) for c in camera_ids]
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.history_frames = int(history_frames)
        self.future_frames = int(future_frames)
        self.total_frames = self.history_frames + self.future_frames
        self.frame_stride = int(frame_stride)
        self.window_stride = int(window_stride)
        self.pose_scale = float(pose_scale)
        self.patch_size = int(patch_size)
        self.samples: list[tuple[Path, list[int]]] = []
        self.scene_cache: dict[Path, dict[str, Any]] = {}

        dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        if split == "train":
            dirs = [p for p in dirs if p.name.startswith("segment-")]
        else:
            dirs = [p for p in dirs if not p.name.startswith("segment-")]
        if max_scenes and max_scenes > 0:
            dirs = dirs[: int(max_scenes)]

        for scene in dirs:
            images_dir = scene / "images"
            ego_dir = scene / "ego_pose"
            if not images_dir.is_dir() or not ego_dir.is_dir():
                continue
            frame_to_cams: dict[int, set[str]] = {}
            for img in images_dir.glob("*.jpg"):
                stem = img.stem
                if "_" not in stem:
                    continue
                frame_s, cam_s = stem.rsplit("_", 1)
                if cam_s not in self.camera_ids:
                    continue
                try:
                    frame_i = int(frame_s)
                except ValueError:
                    continue
                frame_to_cams.setdefault(frame_i, set()).add(cam_s)
            valid = [
                f
                for f, cams in sorted(frame_to_cams.items())
                if all(c in cams for c in self.camera_ids) and (ego_dir / f"{f:03d}.txt").is_file()
            ]
            max_start = len(valid) - (self.total_frames - 1) * self.frame_stride
            for start in range(0, max(0, max_start), max(1, self.window_stride)):
                frames = [valid[start + i * self.frame_stride] for i in range(self.total_frames)]
                self.samples.append((scene, frames))
        if not self.samples:
            raise RuntimeError(f"No Waymo rig windows for split={split} under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_scene_meta(self, scene: Path) -> dict[str, Any]:
        cached = self.scene_cache.get(scene)
        if cached is not None:
            return cached
        extrinsics = {}
        intrinsics = {}
        for cam in self.camera_ids:
            extrinsics[cam] = load_txt_matrix(scene / "extrinsics" / f"{cam}.txt")
            intrinsics[cam] = np.loadtxt(scene / "intrinsics" / f"{cam}.txt", dtype=np.float32)
        cached = {"extrinsics": extrinsics, "intrinsics": intrinsics}
        self.scene_cache[scene] = cached
        return cached

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scene, frames = self.samples[idx]
        meta = self._load_scene_meta(scene)
        h, w = self.image_size
        images_01 = []
        images_norm = []
        intrinsics_out = []
        c2ref_out = []
        ego_ref = load_txt_matrix(scene / "ego_pose" / f"{frames[self.history_frames - 1]:03d}.txt")
        ref_inv = np.linalg.inv(ego_ref).astype(np.float32)

        for frame in frames:
            ego = load_txt_matrix(scene / "ego_pose" / f"{frame:03d}.txt")
            frame_imgs_01 = []
            frame_imgs_norm = []
            frame_intr = []
            frame_c2ref = []
            for cam in self.camera_ids:
                path = scene / "images" / f"{frame:03d}_{cam}.jpg"
                img = Image.open(path).convert("RGB")
                src_size = img.size
                img = img.resize((w, h), resample=getattr(Image, "Resampling", Image).BICUBIC)
                arr = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).permute(2, 0, 1)
                norm = (arr - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
                intr = scale_intrinsics(meta["intrinsics"][cam], src_size=src_size, dst_size=self.image_size)
                c2ref = ref_inv @ ego @ meta["extrinsics"][cam]
                frame_imgs_01.append(arr)
                frame_imgs_norm.append(norm)
                frame_intr.append(torch.from_numpy(intr))
                frame_c2ref.append(torch.from_numpy(c2ref.astype(np.float32)))
            images_01.append(torch.stack(frame_imgs_01, dim=0))
            images_norm.append(torch.stack(frame_imgs_norm, dim=0))
            intrinsics_out.append(torch.stack(frame_intr, dim=0))
            c2ref_out.append(torch.stack(frame_c2ref, dim=0))

        images_01_t = torch.stack(images_01, dim=0)
        images_norm_t = torch.stack(images_norm, dim=0)
        intr_t = torch.stack(intrinsics_out, dim=0)
        c2ref_t = torch.stack(c2ref_out, dim=0)

        rays = []
        for ti in range(c2ref_t.shape[0]):
            rv = []
            for vi in range(c2ref_t.shape[1]):
                rv.append(make_plucker_rays(c2ref_t[ti, vi], intr_t[ti, vi], self.image_size, self.patch_size, self.pose_scale))
            rays.append(torch.stack(rv, dim=0))
        rays_t = torch.stack(rays, dim=0)

        ego_rel = []
        for frame in frames:
            ego = torch.from_numpy((ref_inv @ load_txt_matrix(scene / "ego_pose" / f"{frame:03d}.txt")).astype(np.float32))
            ego_rel.append(torch.stack([ego[0, 3], ego[1, 3], yaw_from_matrix(ego)]))

        return {
            "images": images_norm_t,
            "images_01": images_01_t,
            "rays": rays_t,
            "c2ref": c2ref_t,
            "intrinsics": intr_t,
            "ego_xytheta": torch.stack(ego_rel, dim=0),
            "scene": scene.name,
            "frames": torch.tensor(frames, dtype=torch.long),
        }


class FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(self.q_norm(q), self.kv_norm(kv), self.kv_norm(kv), need_weights=False)
        q = q + out
        q = q + self.ff(self.ff_norm(q))
        return q


class SelfBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        x = x + self.ff(self.norm2(x))
        return x


class FactorizedDenoiseBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.spatial = SelfBlock(dim, heads, mlp_ratio, dropout)
        self.cross_view = SelfBlock(dim, heads, mlp_ratio, dropout)
        self.temporal = SelfBlock(dim, heads, mlp_ratio, dropout)
        self.global_cross = CrossBlock(dim, heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, global_tokens: torch.Tensor) -> torch.Tensor:
        b, t, v, n, d = x.shape
        x = self.spatial(x.reshape(b * t * v, n, d)).reshape(b, t, v, n, d)
        x = x.permute(0, 1, 3, 2, 4).reshape(b * t * n, v, d)
        x = self.cross_view(x).reshape(b, t, n, v, d).permute(0, 1, 3, 2, 4)
        x = x.permute(0, 2, 3, 1, 4).reshape(b * v * n, t, d)
        x = self.temporal(x).reshape(b, v, n, t, d).permute(0, 3, 1, 2, 4)
        flat = x.reshape(b, t * v * n, d)
        flat = self.global_cross(flat, global_tokens)
        x = flat.reshape(b, t, v, n, d)
        return x + self.ff(self.norm(x))


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ARGlobalDenoiser(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_heads: int,
        depth: int,
        global_tokens: int,
        global_layers: int,
        mlp_ratio: float,
        dropout: float,
        max_frames: int = 64,
        max_views: int = 8,
        max_patches: int = 2048,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.in_proj = nn.Linear(feature_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, feature_dim)
        self.ray_mlp = nn.Sequential(nn.Linear(6, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.timestep_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.SiLU(), nn.Linear(hidden_size * 4, hidden_size))
        self.frame_embed = nn.Embedding(max_frames, hidden_size)
        self.view_embed = nn.Embedding(max_views, hidden_size)
        self.patch_embed = nn.Embedding(max_patches, hidden_size)
        self.global_queries = nn.Parameter(torch.randn(global_tokens, hidden_size) * 0.02)
        self.global_encoder = nn.ModuleList([CrossBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(global_layers)])
        self.blocks = nn.ModuleList([FactorizedDenoiseBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.global_update = nn.ModuleList([CrossBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(max(1, global_layers))])
        self.update_gate = nn.Parameter(torch.tensor(-1.0))

    def token_embed(self, z: torch.Tensor, rays: torch.Tensor, frame_offset: int = 0) -> torch.Tensor:
        b, t, v, n, _ = z.shape
        device = z.device
        frame_ids = torch.arange(frame_offset, frame_offset + t, device=device)
        view_ids = torch.arange(v, device=device)
        patch_ids = torch.arange(n, device=device)
        frame_e = self.frame_embed(frame_ids).view(1, t, 1, 1, self.hidden_size)
        view_e = self.view_embed(view_ids).view(1, 1, v, 1, self.hidden_size)
        patch_e = self.patch_embed(patch_ids).view(1, 1, 1, n, self.hidden_size)
        return (
            self.in_proj(z)
            + self.ray_mlp(rays)
            + frame_e
            + view_e
            + patch_e
        )

    def encode_history(self, history_z: torch.Tensor, history_rays: torch.Tensor) -> torch.Tensor:
        b = history_z.shape[0]
        tokens = self.token_embed(history_z, history_rays).reshape(b, -1, self.hidden_size)
        g = self.global_queries.unsqueeze(0).expand(b, -1, -1)
        for block in self.global_encoder:
            g = block(g, tokens)
        return g

    def update_global(self, global_tokens: torch.Tensor, z_clean: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embed(z_clean, rays).reshape(global_tokens.shape[0], -1, self.hidden_size)
        g_new = global_tokens
        for block in self.global_update:
            g_new = block(g_new, tokens)
        gate = torch.sigmoid(self.update_gate)
        return global_tokens * (1.0 - gate) + g_new * gate

    def forward(
        self,
        noisy_z: torch.Tensor,
        future_rays: torch.Tensor,
        global_tokens: torch.Tensor,
        t: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(noisy_z, future_rays)
        time_emb = self.timestep_mlp(sinusoidal_embedding(t.reshape(-1).to(x.dtype), self.hidden_size)).view(x.shape[0], 1, 1, 1, -1)
        x = x + time_emb
        for block in self.blocks:
            x = block(x, global_tokens)
        return self.out_proj(x)

    def training_losses(
        self,
        target_z: torch.Tensor,
        future_rays: torch.Tensor,
        global_tokens: torch.Tensor,
        global_drop: float = 0.0,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        noise = torch.randn_like(target_z)
        t = torch.rand((target_z.shape[0],), device=target_z.device, dtype=target_z.dtype).clamp(1e-4, 1.0)
        x_t = t.view(-1, 1, 1, 1, 1) * target_z + (1.0 - t).view(-1, 1, 1, 1, 1) * noise
        target_v = target_z - noise
        if self.training and global_drop > 0:
            keep = (torch.rand((global_tokens.shape[0], 1, 1), device=global_tokens.device) > global_drop).to(global_tokens.dtype)
            global_tokens = global_tokens * keep
        pred_v = self(x_t, future_rays, global_tokens, t, actions=actions)
        pred_clean = x_t + pred_v * (1.0 - t).view(-1, 1, 1, 1, 1)
        mse = (pred_v.float() - target_v.float()).pow(2).mean()
        return {"loss": mse, "pred_clean": pred_clean.detach(), "pred_v": pred_v}

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int, int, int],
        future_rays: torch.Tensor,
        global_tokens: torch.Tensor,
        steps: int,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = torch.randn(shape, device=future_rays.device, dtype=future_rays.dtype)
        ts = torch.linspace(0.0, 1.0, int(steps) + 1, device=future_rays.device, dtype=future_rays.dtype)
        for curr, nxt in zip(ts[:-1], ts[1:]):
            t_vec = curr.expand(shape[0])
            v = self(z, future_rays, global_tokens, t_vec, actions=actions)
            z = z + (nxt - curr) * v
        return z


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class AdaLNCDiTBlock(nn.Module):
    """NWM-style target self-attention plus cross-attention to world context."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, mlp_ratio, dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 9))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_xa,
            scale_xa,
            gate_xa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada(cond).chunk(9, dim=-1)
        q = modulate(self.norm1(x), shift_msa, scale_msa)
        y, _ = self.self_attn(q, q, q, need_weights=False)
        x = x + gate_msa[:, None, :] * y

        q = modulate(self.norm2(x), shift_xa, scale_xa)
        kv = self.context_norm(context)
        y, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + gate_xa[:, None, :] * y

        y = self.ff(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x + gate_mlp[:, None, :] * y


class ARCDiTDenoiser(nn.Module):
    """Formal autoregressive cDiT over dense multi-view geometry latents.

    History tokens are compressed into recurrent world tokens with Perceiver-style
    cross-attention. Each future AR step jointly denoises all target-view tokens:
    target self-attention handles within-frame/view consistency, cross-attention
    reads the shared world tokens, and AdaLN injects action and diffusion time.
    """

    uses_actions = True

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int,
        num_heads: int,
        depth: int,
        global_tokens: int,
        global_layers: int,
        mlp_ratio: float,
        dropout: float,
        max_frames: int = 64,
        max_views: int = 8,
        max_patches: int = 2048,
        action_dim: int = 4,
        global_update_layers: int | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        update_layers = int(global_update_layers if global_update_layers is not None else max(1, global_layers // 2))

        self.target_in_proj = nn.Linear(feature_dim, hidden_size)
        self.context_in_proj = nn.Linear(feature_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, feature_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.ray_mlp = nn.Sequential(nn.Linear(6, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.action_token_mlp = nn.Sequential(nn.Linear(action_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.action_cond_mlp = nn.Sequential(nn.Linear(action_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.timestep_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.SiLU(), nn.Linear(hidden_size * 4, hidden_size))

        self.frame_embed = nn.Embedding(max_frames, hidden_size)
        self.view_embed = nn.Embedding(max_views, hidden_size)
        self.patch_embed = nn.Embedding(max_patches, hidden_size)

        self.global_queries = nn.Parameter(torch.randn(global_tokens, hidden_size) * 0.02)
        self.global_cross_blocks = nn.ModuleList(
            [CrossBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(global_layers)]
        )
        self.global_self_blocks = nn.ModuleList(
            [SelfBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(global_layers)]
        )
        self.blocks = nn.ModuleList([AdaLNCDiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.update_cross_blocks = nn.ModuleList(
            [CrossBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(update_layers)]
        )
        self.update_self_blocks = nn.ModuleList(
            [SelfBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(update_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.update_gate = nn.Parameter(torch.tensor(-1.0))

    def _pos_embed(self, b: int, t: int, v: int, n: int, device: torch.device) -> torch.Tensor:
        frame_ids = torch.arange(t, device=device).clamp_max(self.frame_embed.num_embeddings - 1)
        view_ids = torch.arange(v, device=device).clamp_max(self.view_embed.num_embeddings - 1)
        patch_ids = torch.arange(n, device=device).clamp_max(self.patch_embed.num_embeddings - 1)
        frame_e = self.frame_embed(frame_ids).view(1, t, 1, 1, self.hidden_size)
        view_e = self.view_embed(view_ids).view(1, 1, v, 1, self.hidden_size)
        patch_e = self.patch_embed(patch_ids).view(1, 1, 1, n, self.hidden_size)
        return (frame_e + view_e + patch_e).expand(b, -1, -1, -1, -1)

    def _context_token_embed(self, z: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        b, t, v, n, _ = z.shape
        return self.context_in_proj(z) + self.ray_mlp(rays) + self._pos_embed(b, t, v, n, z.device)

    def _target_token_embed(self, z: torch.Tensor, rays: torch.Tensor, actions: torch.Tensor | None) -> torch.Tensor:
        b, t, v, n, _ = z.shape
        x = self.target_in_proj(z) + self.ray_mlp(rays) + self._pos_embed(b, t, v, n, z.device)
        if actions is not None:
            x = x + self.action_token_mlp(actions.to(device=z.device, dtype=z.dtype)).view(b, t, 1, 1, self.hidden_size)
        return x

    def _cond_embed(self, t: torch.Tensor, actions: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
        t_emb = self.timestep_mlp(sinusoidal_embedding(t.reshape(-1).to(dtype), self.hidden_size))
        if actions is None:
            return t_emb
        action_emb = self.action_cond_mlp(actions.to(device=t.device, dtype=dtype).mean(dim=1))
        return t_emb + action_emb

    def _maybe_checkpoint(self, fn, *args):
        if self.training and self.gradient_checkpointing:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def encode_history(self, history_z: torch.Tensor, history_rays: torch.Tensor) -> torch.Tensor:
        b = history_z.shape[0]
        tokens = self._context_token_embed(history_z, history_rays).reshape(b, -1, self.hidden_size)
        g = self.global_queries.unsqueeze(0).expand(b, -1, -1)
        for cross, self_block in zip(self.global_cross_blocks, self.global_self_blocks):
            g = self._maybe_checkpoint(cross, g, tokens)
            g = self._maybe_checkpoint(self_block, g)
        return g

    def update_global(
        self,
        global_tokens: torch.Tensor,
        z_clean: torch.Tensor,
        rays: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self._context_token_embed(z_clean, rays).reshape(global_tokens.shape[0], -1, self.hidden_size)
        g_new = global_tokens
        for cross, self_block in zip(self.update_cross_blocks, self.update_self_blocks):
            g_new = self._maybe_checkpoint(cross, g_new, tokens)
            g_new = self._maybe_checkpoint(self_block, g_new)
        gate = torch.sigmoid(self.update_gate)
        return global_tokens * (1.0 - gate) + g_new * gate

    def forward(
        self,
        noisy_z: torch.Tensor,
        future_rays: torch.Tensor,
        global_tokens: torch.Tensor,
        t: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, tf, v, n, _ = noisy_z.shape
        x = self._target_token_embed(noisy_z, future_rays, actions).reshape(b, tf * v * n, self.hidden_size)
        cond = self._cond_embed(t, actions, x.dtype)
        context = global_tokens
        for block in self.blocks:
            x = self._maybe_checkpoint(block, x, context, cond)
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.out_proj(x).reshape(b, tf, v, n, self.feature_dim)

    def training_losses(
        self,
        target_z: torch.Tensor,
        future_rays: torch.Tensor,
        global_tokens: torch.Tensor,
        global_drop: float = 0.0,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        noise = torch.randn_like(target_z)
        t = torch.rand((target_z.shape[0],), device=target_z.device, dtype=target_z.dtype).clamp(1e-4, 1.0)
        x_t = t.view(-1, 1, 1, 1, 1) * target_z + (1.0 - t).view(-1, 1, 1, 1, 1) * noise
        target_v = target_z - noise
        if self.training and global_drop > 0:
            keep = (torch.rand((global_tokens.shape[0], 1, 1), device=global_tokens.device) > global_drop).to(global_tokens.dtype)
            global_tokens = global_tokens * keep
        pred_v = self(x_t, future_rays, global_tokens, t, actions=actions)
        pred_clean = x_t + pred_v * (1.0 - t).view(-1, 1, 1, 1, 1)
        mse = (pred_v.float() - target_v.float()).pow(2).mean()
        return {"loss": mse, "pred_clean": pred_clean.detach(), "pred_v": pred_v}

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int, int, int],
        future_rays: torch.Tensor,
        global_tokens: torch.Tensor,
        steps: int,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = torch.randn(shape, device=future_rays.device, dtype=future_rays.dtype)
        ts = torch.linspace(0.0, 1.0, int(steps) + 1, device=future_rays.device, dtype=future_rays.dtype)
        for curr, nxt in zip(ts[:-1], ts[1:]):
            t_vec = curr.expand(shape[0])
            v = self(z, future_rays, global_tokens, t_vec, actions=actions)
            z = z + (nxt - curr) * v
        return z


def maybe_drop_history(z: torch.Tensor, rays: torch.Tensor, camera_drop: float, frame_drop: float) -> tuple[torch.Tensor, torch.Tensor]:
    if camera_drop <= 0 and frame_drop <= 0:
        return z, rays
    b, t, v = z.shape[:3]
    mask = torch.ones((b, t, v, 1, 1), device=z.device, dtype=z.dtype)
    if camera_drop > 0:
        cam_keep = (torch.rand((b, 1, v, 1, 1), device=z.device) > camera_drop).to(z.dtype)
        mask = mask * cam_keep
    if frame_drop > 0:
        frame_keep = (torch.rand((b, t, 1, 1, 1), device=z.device) > frame_drop).to(z.dtype)
        mask = mask * frame_keep
    return z * mask, rays * mask


def maybe_perturb_rays(rays: torch.Tensor, drop: float, noise_std: float) -> torch.Tensor:
    if drop > 0:
        keep = (torch.rand((rays.shape[0], rays.shape[1], rays.shape[2], 1, 1), device=rays.device) > drop).to(rays.dtype)
        rays = rays * keep
    if noise_std > 0:
        rays = rays + torch.randn_like(rays) * float(noise_std)
    return rays


def extract_da3_tokens(rae, images_norm: torch.Tensor, precision: str) -> torch.Tensor:
    b, t, v, c, h, w = images_norm.shape
    flat = images_norm.reshape(b, t * v, c, h, w)
    use_amp = precision in {"bf16", "fp16"} and flat.is_cuda
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad(), torch.autocast(device_type=flat.device.type, dtype=amp_dtype, enabled=use_amp):
        feats = rae.encode(flat, mode="all")
        z = torch.cat([feats[k][:, 1:, :] for k in sorted(feats.keys())], dim=-1)
    return z.reshape(b, t, v, z.shape[-2], z.shape[-1]).contiguous()


def extract_da3_tokens_chunked(rae, images_norm: torch.Tensor, precision: str, chunk_size: int = 0) -> torch.Tensor:
    if chunk_size is None or int(chunk_size) <= 0 or images_norm.shape[0] <= int(chunk_size):
        return extract_da3_tokens(rae, images_norm, precision)
    outs = []
    for start in range(0, images_norm.shape[0], int(chunk_size)):
        outs.append(extract_da3_tokens(rae, images_norm[start : start + int(chunk_size)], precision))
    return torch.cat(outs, dim=0)


class FeatureNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        self.mean = mean.view(1, 1, 1, 1, -1)
        self.std = std.clamp_min(float(eps)).view(1, 1, 1, 1, -1)

    @classmethod
    def identity(cls, feature_dim: int, device: torch.device) -> "FeatureNormalizer":
        return cls(torch.zeros(feature_dim, device=device), torch.ones(feature_dim, device=device))

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "FeatureNormalizer":
        obj = torch.load(str(path), map_location=device)
        return cls(obj["mean"].to(device), obj["std"].to(device), float(obj.get("eps", 1e-6)))

    @classmethod
    def from_checkpoint(cls, ckpt: dict[str, Any], feature_dim: int, device: torch.device) -> "FeatureNormalizer":
        obj = ckpt.get("normalizer")
        if obj is None:
            return cls.identity(feature_dim, device)
        return cls(obj["mean"].to(device), obj["std"].to(device), float(obj.get("eps", 1e-6)))

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {"mean": self.mean.reshape(-1).detach().cpu(), "std": self.std.reshape(-1).detach().cpu(), "eps": 1e-6}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), str(path))

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean.to(z.device, z.dtype)) / self.std.to(z.device, z.dtype)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z.device, z.dtype) + self.mean.to(z.device, z.dtype)


@torch.no_grad()
def fit_or_load_feature_normalizer(
    cfg,
    train_loader: DataLoader,
    rae,
    device: torch.device,
    precision: str,
    out_dir: Path,
    *,
    distributed: bool = False,
    rank: int = 0,
) -> FeatureNormalizer:
    norm_cfg = cfg.get("normalization", {})
    enabled = bool(norm_cfg.get("enabled", False))
    stats_path = Path(str(norm_cfg.get("stats_path", out_dir / "normalizer_stats.pt")))
    if not stats_path.is_absolute():
        stats_path = out_dir / stats_path
    feature_dim = int(cfg.da3.feature_dim)
    if not enabled:
        return FeatureNormalizer.identity(feature_dim, device)
    if stats_path.is_file() and not bool(norm_cfg.get("recompute", False)):
        rank0_print(rank, json.dumps({"normalizer": "loaded", "path": str(stats_path)}, ensure_ascii=False))
        return FeatureNormalizer.load(stats_path, device)

    max_batches = int(norm_cfg.get("num_batches", 256))
    encode_chunk_size = int(cfg.da3.get("encode_chunk_size", 0))
    sum_x = torch.zeros(feature_dim, device=device, dtype=torch.float64)
    sum_x2 = torch.zeros(feature_dim, device=device, dtype=torch.float64)
    count = 0
    seen = 0
    for batch in tqdm(train_loader, total=max_batches, desc="fit-normalizer", disable=not is_main_process(rank)):
        images = batch["images"].to(device, non_blocking=True)
        z = extract_da3_tokens_chunked(rae, images, precision, encode_chunk_size).float()
        flat = z.reshape(-1, z.shape[-1]).to(torch.float64)
        sum_x += flat.sum(dim=0)
        sum_x2 += flat.square().sum(dim=0)
        count += flat.shape[0]
        seen += 1
        if seen >= max_batches:
            break
    count_t = torch.tensor([count], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_x2, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    count = int(count_t.item())
    mean = (sum_x / max(count, 1)).float()
    var = (sum_x2 / max(count, 1)).float() - mean.square()
    std = var.clamp_min(float(norm_cfg.get("eps", 1e-6))).sqrt()
    normalizer = FeatureNormalizer(mean, std, float(norm_cfg.get("eps", 1e-6)))
    if is_main_process(rank):
        normalizer.save(stats_path)
        print(json.dumps({"normalizer": "fitted", "path": str(stats_path), "batches_per_rank": seen, "count": count, "std_mean": float(std.mean().cpu())}, ensure_ascii=False))
    if distributed:
        dist.barrier()
    return normalizer


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj) and torch.is_tensor(next(iter(obj.values())))


def _strip_module_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not sd:
        return sd
    if next(iter(sd)).startswith("module."):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def extract_decoder_state(ckpt_obj: Any, prefer_ema: bool) -> dict[str, torch.Tensor]:
    if _looks_like_state_dict(ckpt_obj):
        return _strip_module_prefix(ckpt_obj)
    if prefer_ema and isinstance(ckpt_obj, dict) and "ema_decoder" in ckpt_obj:
        return _strip_module_prefix(ckpt_obj["ema_decoder"])
    if isinstance(ckpt_obj, dict) and "decoder" in ckpt_obj:
        return _strip_module_prefix(ckpt_obj["decoder"])
    if isinstance(ckpt_obj, dict) and "ema_decoder" in ckpt_obj:
        return _strip_module_prefix(ckpt_obj["ema_decoder"])
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
        return _strip_module_prefix(ckpt_obj["model"])
    raise ValueError("Cannot find decoder state in checkpoint")


def build_rae_and_decoder(cfg, device: torch.device):
    from stage1.rae_da3 import RAE_DA3
    from stage1.decoders import GeneralDecoder_Variable

    da3 = cfg.da3
    rae = RAE_DA3(
        encoder_pretrained_path=str(da3.encoder_pretrained_path),
        encoder_input_size=504,
        encoder_type="DA3EncoderDirect",
        reshape_to_2d=True,
    ).to(device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad_(False)

    dec_cfg = AutoConfig.from_pretrained(str(da3.decoder_config_path))
    dec_cfg.hidden_size = int(da3.feature_dim)
    dec_cfg.patch_size = int(da3.patch_size)
    decoder = GeneralDecoder_Variable(dec_cfg, base_image_size=(504, 504)).to(device)
    ckpt = torch.load(str(da3.rgb_decoder_ckpt), map_location="cpu")
    decoder.load_state_dict(extract_decoder_state(ckpt, bool(da3.get("use_ema_decoder", True))), strict=True)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return rae, decoder


def decode_rgb(decoder, z: torch.Tensor, image_size: tuple[int, int], precision: str, allow_grad: bool = False) -> torch.Tensor:
    b, t, v, n, c = z.shape
    h, w = image_size
    flat = z.reshape(b * t * v, n, c)
    use_amp = precision in {"bf16", "fp16"} and flat.is_cuda
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.set_grad_enabled(allow_grad), torch.autocast(device_type=flat.device.type, dtype=amp_dtype, enabled=use_amp):
        out = decoder(flat, input_size=(h, w), drop_cls_token=False).logits
        rec = decoder.unpatchify(out, (h, w))
    rec = rec * IMAGENET_STD.to(rec.device, rec.dtype) + IMAGENET_MEAN.to(rec.device, rec.dtype)
    return rec.reshape(b, t, v, 3, h, w).clamp(0, 1)


def save_rgb_grid(path: Path, history: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor) -> None:
    # tensors: history (Th,V,3,H,W), gt/pred (Tf,V,3,H,W), all [0,1]
    tiles = []
    for seq in [history, gt, pred]:
        row_frames = []
        for ti in range(seq.shape[0]):
            cams = [seq[ti, vi].detach().float().cpu().permute(1, 2, 0).numpy() for vi in range(seq.shape[1])]
            row_frames.append(np.concatenate(cams, axis=1))
        tiles.append(np.concatenate(row_frames, axis=0))
    max_h = max(tile.shape[0] for tile in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[0] < max_h:
            pad = np.zeros((max_h - tile.shape[0], tile.shape[1], tile.shape[2]), dtype=tile.dtype)
            tile = np.concatenate([tile, pad], axis=0)
        padded.append(tile)
    canvas = np.concatenate(padded, axis=1)
    Image.fromarray((canvas.clip(0, 1) * 255).astype(np.uint8)).save(path)


def save_depth_proxy(path: Path, z: torch.Tensor, image_size: tuple[int, int]) -> None:
    b, t, v, n, c = z.shape
    h, w = image_size
    hp = h // 14
    wp = w // 14
    proxy = z.float().pow(2).mean(dim=-1).sqrt().reshape(b, t, v, hp, wp)
    proxy = F.interpolate(proxy.reshape(b * t * v, 1, hp, wp), size=(h, w), mode="bilinear", align_corners=False)
    proxy = proxy.reshape(b, t, v, h, w)[0]
    imgs = []
    for ti in range(t):
        cams = []
        for vi in range(v):
            arr = proxy[ti, vi].detach().cpu().numpy()
            arr = (arr - np.percentile(arr, 2)) / (np.percentile(arr, 98) - np.percentile(arr, 2) + 1e-6)
            cams.append(np.repeat(arr.clip(0, 1)[..., None], 3, axis=-1))
        imgs.append(np.concatenate(cams, axis=1))
    Image.fromarray((np.concatenate(imgs, axis=0) * 255).astype(np.uint8)).save(path)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vals = depth[finite]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    # Compact magma-like ramp without depending on matplotlib.
    stops = np.array(
        [
            [0.001, 0.000, 0.014],
            [0.251, 0.059, 0.416],
            [0.578, 0.148, 0.404],
            [0.865, 0.318, 0.226],
            [0.988, 0.647, 0.039],
            [0.987, 0.991, 0.749],
        ],
        dtype=np.float32,
    )
    pos = x * (len(stops) - 1)
    i0 = np.floor(pos).astype(np.int64).clip(0, len(stops) - 1)
    i1 = np.ceil(pos).astype(np.int64).clip(0, len(stops) - 1)
    a = (pos - i0)[..., None]
    rgb = stops[i0] * (1.0 - a) + stops[i1] * a
    return (rgb * 255).astype(np.uint8)


def save_depth_grid(path: Path, depth: torch.Tensor) -> None:
    # depth: (T,V,H,W)
    rows = []
    for ti in range(depth.shape[0]):
        cams = [colorize_depth(depth[ti, vi].detach().float().cpu().numpy()) for vi in range(depth.shape[1])]
        rows.append(np.concatenate(cams, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


@torch.no_grad()
def predict_da3_depth(rgb: torch.Tensor, model_path: str, device: torch.device, precision: str) -> torch.Tensor:
    from depth_anything_3.api import DepthAnything3

    b, t, v, c, h, w = rgb.shape
    model = DepthAnything3.from_pretrained(model_path).to(device).eval()
    flat = rgb.reshape(b, t * v, c, h, w).to(device)
    use_amp = precision in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        out = model(flat, export_feat_layers=[5, 7, 9, 11])
    depth = out["depth"]
    if depth.ndim == 4:
        depth = depth.reshape(b, t, v, h, w)
    elif depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth.squeeze(2).reshape(b, t, v, h, w)
    else:
        depth = depth.reshape(b, t, v, h, w)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return depth


def make_action_condition(ego_xytheta: torch.Tensor, pose_scale: float) -> torch.Tensor:
    """Metric action conditioning relative to the history-last ego reference."""
    xy = ego_xytheta[..., :2] / float(pose_scale)
    theta = ego_xytheta[..., 2:3]
    return torch.cat([xy, torch.sin(theta), torch.cos(theta)], dim=-1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_denoiser(cfg) -> nn.Module:
    model_cfg = cfg.model
    architecture = str(model_cfg.get("architecture", "factorized_global")).lower()
    common = dict(
        feature_dim=int(cfg.da3.feature_dim),
        hidden_size=int(model_cfg.hidden_size),
        num_heads=int(model_cfg.num_heads),
        depth=int(model_cfg.depth),
        global_tokens=int(model_cfg.global_tokens),
        global_layers=int(model_cfg.global_layers),
        mlp_ratio=float(model_cfg.mlp_ratio),
        dropout=float(model_cfg.dropout),
        max_views=max(8, len(cfg.data.camera_ids)),
        max_patches=(int(cfg.data.image_size[0]) // int(cfg.da3.patch_size))
        * (int(cfg.data.image_size[1]) // int(cfg.da3.patch_size)),
    )
    if architecture in {"cdit", "cdit_formal", "dual_stream_cdit", "formal"}:
        return ARCDiTDenoiser(
            **common,
            action_dim=4,
            global_update_layers=int(model_cfg.get("global_update_layers", max(1, int(model_cfg.global_layers) // 2))),
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", False)),
        )
    if architecture not in {"factorized_global", "small", "prototype"}:
        raise ValueError(f"Unknown model architecture: {architecture}")
    return ARGlobalDenoiser(**common)


class ARTrainingStep(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        global_condition_drop: float,
        chain_forward_prob: float,
        scheduled_sampling_max: float,
        scheduled_sampling_warmup_steps: int,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.global_condition_drop = float(global_condition_drop)
        self.chain_forward_prob = float(chain_forward_prob)
        self.scheduled_sampling_max = float(scheduled_sampling_max)
        self.scheduled_sampling_warmup_steps = int(max(scheduled_sampling_warmup_steps, 1))

    def forward(
        self,
        hist_z: torch.Tensor,
        hist_rays: torch.Tensor,
        fut_z: torch.Tensor,
        fut_rays_cond: torch.Tensor,
        fut_rays_clean: torch.Tensor,
        fut_actions: torch.Tensor | None,
        step_value: int | float,
    ) -> dict[str, torch.Tensor]:
        fut = int(fut_z.shape[1])
        g = self.denoiser.encode_history(hist_z, hist_rays)
        step_f = float(step_value)
        ss_prob = min(self.scheduled_sampling_max, self.scheduled_sampling_max * step_f / self.scheduled_sampling_warmup_steps)

        if fut == 1 or random.random() < self.chain_forward_prob:
            losses = []
            pred_for_rgb = []
            for ti in range(fut):
                cur_action = fut_actions[:, ti : ti + 1] if fut_actions is not None else None
                terms = self.denoiser.training_losses(
                    fut_z[:, ti : ti + 1],
                    fut_rays_cond[:, ti : ti + 1],
                    g,
                    self.global_condition_drop,
                    actions=cur_action,
                )
                losses.append(terms["loss"])
                pred_clean = terms["pred_clean"]
                pred_for_rgb.append(pred_clean)
                update_source = pred_clean if random.random() < ss_prob else fut_z[:, ti : ti + 1]
                g = self.denoiser.update_global(g, update_source.detach(), fut_rays_clean[:, ti : ti + 1], actions=cur_action)
            feature_loss = torch.stack(losses).mean()
            pred_clean_all = torch.cat(pred_for_rgb, dim=1)
        else:
            terms = self.denoiser.training_losses(fut_z, fut_rays_cond, g, self.global_condition_drop, actions=fut_actions)
            feature_loss = terms["loss"]
            pred_clean_all = terms["pred_clean"]

        return {"feature_loss": feature_loss, "pred_clean": pred_clean_all}


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        import copy

        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.model.state_dict().items():
            if torch.is_floating_point(v):
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


def build_loaders(
    cfg,
    args,
    *,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    data = cfg.data
    train_ds = WaymoRigWindowDataset(
        root=str(data.root),
        split="train",
        camera_ids=list(data.camera_ids),
        image_size=tuple(data.image_size),
        history_frames=int(data.history_frames),
        future_frames=int(data.future_frames),
        frame_stride=int(data.frame_stride),
        window_stride=int(data.train_window_stride),
        pose_scale=float(data.pose_scale),
        max_scenes=int(data.get("max_train_scenes", 0)),
        patch_size=int(cfg.da3.patch_size),
    )
    val_ds = WaymoRigWindowDataset(
        root=str(data.root),
        split="val",
        camera_ids=list(data.camera_ids),
        image_size=tuple(data.image_size),
        history_frames=int(data.history_frames),
        future_frames=max(int(data.future_frames), int(cfg.rollout.get("frames", 20))),
        frame_stride=int(data.frame_stride),
        window_stride=int(data.val_window_stride),
        pose_scale=float(data.pose_scale),
        max_scenes=int(data.get("max_val_scenes", 1)),
        patch_size=int(cfg.da3.patch_size),
    )
    global_batch_size = int(args.batch_size or cfg.training.batch_size)
    if distributed:
        if global_batch_size % world_size != 0:
            raise ValueError(f"Global batch_size={global_batch_size} must be divisible by world_size={world_size} for DDP.")
        batch_size = global_batch_size // world_size
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg.training.seed),
            drop_last=True,
        )
        shuffle = False
    else:
        batch_size = global_batch_size
        train_sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=int(args.num_workers if args.num_workers is not None else cfg.training.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    rank0_print(
        rank,
        json.dumps(
            {
                "loaded_waymo_rig_windows": {"train": len(train_ds), "val": len(val_ds)},
                "global_batch_size": global_batch_size,
                "per_rank_batch_size": batch_size,
                "world_size": world_size,
            },
            ensure_ascii=False,
        ),
    )
    return train_loader, val_loader


def cfg_get(args, cfg, path: str):
    cur = cfg
    for p in path.split("."):
        cur = cur[p]
    return cur


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def plot_loss_curves(metrics_path: Path, out_path: Path) -> None:
    if not metrics_path.is_file():
        return
    train_steps, train_loss, rgb_steps, rgb_loss = [], [], [], []
    val_steps, val_feature, val_rgb = [], [], []
    for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("split") == "train":
            train_steps.append(obj["step"])
            train_loss.append(obj["feature_loss"])
            if obj.get("rgb_loss", 0.0) > 0:
                rgb_steps.append(obj["step"])
                rgb_loss.append(obj["rgb_loss"])
        elif obj.get("split") == "val":
            val_steps.append(obj["step"])
            val_feature.append(obj["val_feature_mse"])
            val_rgb.append(obj["val_rgb_l1"])
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(train_steps, train_loss, linewidth=1.0)
        axes[0].set_title("train feature loss")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("MSE")
        if val_steps:
            axes[1].plot(val_steps, val_feature, marker=".", linewidth=1.0)
        axes[1].set_title("val feature MSE")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("MSE")
        if rgb_steps:
            axes[2].plot(rgb_steps, rgb_loss, linewidth=1.0, label="train RGB monitor")
        if val_steps:
            axes[2].plot(val_steps, val_rgb, marker=".", linewidth=1.0, label="val RGB monitor")
        axes[2].set_title("RGB monitor only")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("L1")
        axes[2].legend(loc="best")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(json.dumps({"plot_loss_curves": "failed", "error": str(exc)}, ensure_ascii=False))


def save_full_checkpoint(path: Path, model: nn.Module, ema: EMA, optimizer, step: int, cfg, normalizer: FeatureNormalizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def save_inference_checkpoint(path: Path, model: nn.Module, ema: EMA, step: int, cfg, normalizer: FeatureNormalizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "normalizer": normalizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


@torch.no_grad()
def validate_and_visualize(
    model: nn.Module,
    loader: DataLoader,
    rae,
    decoder,
    normalizer: FeatureNormalizer,
    cfg,
    device: torch.device,
    out_dir: Path,
    step: int,
    precision: str,
    sample_steps: int,
) -> dict[str, float]:
    model.eval()
    batch = next(iter(loader))
    images = batch["images"].to(device)
    images_01 = batch["images_01"].to(device)
    rays = batch["rays"].to(device)
    hist = int(cfg.data.history_frames)
    fut = int(cfg.data.future_frames)
    z_raw = extract_da3_tokens_chunked(rae, images[:, : hist + fut], precision, int(cfg.da3.get("encode_chunk_size", 0)))
    z = normalizer.normalize(z_raw)
    hist_z, fut_z = z[:, :hist], z[:, hist : hist + fut]
    hist_rays, fut_rays = rays[:, :hist], rays[:, hist : hist + fut]
    actions = make_action_condition(batch["ego_xytheta"].to(device), float(cfg.data.pose_scale))
    fut_actions = actions[:, hist : hist + fut]
    g = model.encode_history(hist_z, hist_rays)
    pred = []
    for ti in range(fut):
        cur_ray = fut_rays[:, ti : ti + 1]
        cur_action = fut_actions[:, ti : ti + 1] if getattr(model, "uses_actions", False) else None
        cur = model.sample((z.shape[0], 1, z.shape[2], z.shape[3], z.shape[4]), cur_ray, g, sample_steps, actions=cur_action)
        pred.append(cur)
        g = model.update_global(g, cur, cur_ray, actions=cur_action)
    pred_z = torch.cat(pred, dim=1)
    feature_mse = F.mse_loss(pred_z.float(), fut_z.float()).item()
    pred_z_raw = normalizer.denormalize(pred_z)
    pred_rgb = decode_rgb(decoder, pred_z_raw, tuple(cfg.data.image_size), precision)
    gt_rgb = images_01[:, hist : hist + fut]
    rgb_l1 = F.l1_loss(pred_rgb.float(), gt_rgb.float()).item()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rgb_grid(out_dir / f"step_{step:07d}_rgb.png", images_01[0, :hist], gt_rgb[0], pred_rgb[0])
    save_depth_proxy(out_dir / f"step_{step:07d}_depth_proxy.png", pred_z_raw, tuple(cfg.data.image_size))
    model.train()
    return {"val_feature_mse": feature_mse, "val_rgb_l1": rgb_l1}


def train(args, cfg) -> None:
    add_repo_to_path()
    distributed, rank, world_size, local_rank = init_distributed(args)
    seed_everything(int(cfg.training.seed) + rank)
    device = torch.device("cuda", local_rank) if distributed else torch.device(str(args.device or cfg.training.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    out_dir = Path(str(args.output_dir or cfg.project.output_dir))
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    if distributed:
        dist.barrier()

    train_loader, val_loader = build_loaders(cfg, args, distributed=distributed, rank=rank, world_size=world_size)
    rae, decoder = build_rae_and_decoder(cfg, device)
    precision = str(args.precision if args.precision is not None else cfg.training.precision)
    normalizer = fit_or_load_feature_normalizer(
        cfg,
        train_loader,
        rae,
        device,
        precision,
        out_dir,
        distributed=distributed,
        rank=rank,
    )

    model = build_denoiser(cfg).to(device)
    total_params, trainable_params = count_parameters(model)
    summary = {
        "architecture": str(cfg.model.get("architecture", "factorized_global")),
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "fp32_weight_gb": total_params * 4 / (1024**3),
        "normalization": bool(cfg.get("normalization", {}).get("enabled", False)),
    }
    rank0_print(rank, json.dumps({"model_summary": summary}, ensure_ascii=False))
    if is_main_process(rank):
        (out_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if bool(cfg.training.get("compile", False)):
        model = torch.compile(model)
    ema = EMA(model, float(cfg.training.ema_decay))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay), betas=(0.9, 0.95))
    train_step_module: nn.Module = ARTrainingStep(
        model,
        float(cfg.tricks.global_condition_drop),
        float(cfg.tricks.chain_forward_prob),
        float(cfg.tricks.scheduled_sampling_max),
        int(cfg.tricks.scheduled_sampling_warmup_steps),
    ).to(device)
    if distributed:
        train_step_module = DDP(
            train_step_module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=False,
        )
    rank0_print(
        rank,
        json.dumps(
            {
                "distributed_data_parallel": distributed,
                "world_size": world_size,
                "visible_cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0,
                "local_rank": local_rank,
            },
            ensure_ascii=False,
        ),
    )

    max_steps = int(args.max_steps if args.max_steps is not None else cfg.training.max_steps)
    log_every = int(args.log_every if args.log_every is not None else cfg.training.log_every)
    val_every = int(args.val_every if args.val_every is not None else cfg.training.val_every)
    full_ckpt_every = int(args.ckpt_every if args.ckpt_every is not None else cfg.training.get("full_ckpt_every", cfg.training.get("ckpt_every", 0)))
    infer_ckpt_every = int(args.infer_ckpt_every if args.infer_ckpt_every is not None else cfg.training.get("infer_ckpt_every", 0))
    use_amp = precision in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and device.type == "cuda"))
    metrics_path = out_dir / "metrics.jsonl"
    encode_chunk_size = int(cfg.da3.get("encode_chunk_size", 0))

    step = 0
    running = {}
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch = 0
    while step < max_steps:
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            step += 1
            batch_start = time.perf_counter()
            images = batch["images"].to(device, non_blocking=True)
            rays = batch["rays"].to(device, non_blocking=True)
            actions = make_action_condition(batch["ego_xytheta"].to(device, non_blocking=True), float(cfg.data.pose_scale))
            hist = int(cfg.data.history_frames)
            fut = int(cfg.data.future_frames)
            with torch.no_grad():
                z_raw = extract_da3_tokens_chunked(rae, images[:, : hist + fut], precision, encode_chunk_size)
                z = normalizer.normalize(z_raw)
            hist_z, fut_z = z[:, :hist], z[:, hist : hist + fut]
            hist_rays, fut_rays = rays[:, :hist], rays[:, hist : hist + fut]
            fut_actions = actions[:, hist : hist + fut] if getattr(model, "uses_actions", False) else None
            hist_z, hist_rays = maybe_drop_history(hist_z, hist_rays, float(cfg.tricks.history_camera_drop), float(cfg.tricks.history_frame_drop))
            fut_rays_cond = maybe_perturb_rays(fut_rays, float(cfg.tricks.future_ray_drop), float(cfg.tricks.ray_noise_std))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                terms = train_step_module(
                    hist_z,
                    hist_rays,
                    fut_z,
                    fut_rays_cond,
                    fut_rays,
                    fut_actions,
                    step,
                )
                feature_loss = terms["feature_loss"].mean()
                pred_clean_all = terms["pred_clean"]
                loss = feature_loss * float(cfg.loss.feature_weight)
                rgb_loss = torch.tensor(0.0, device=device)
                if float(cfg.loss.rgb_weight) > 0 and step % int(cfg.loss.rgb_loss_every) == 0:
                    with torch.no_grad():
                        pred_rgb = decode_rgb(decoder, normalizer.denormalize(pred_clean_all), tuple(cfg.data.image_size), precision, allow_grad=False)
                        gt_rgb = batch["images_01"].to(device, non_blocking=True)[:, hist : hist + fut]
                        rgb_loss = F.l1_loss(pred_rgb.float(), gt_rgb.float())

            scaler.scale(loss).backward()
            if float(cfg.training.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            elapsed = time.perf_counter() - batch_start
            loss_mean = reduce_mean(loss.detach(), distributed)
            feature_loss_mean = reduce_mean(feature_loss.detach(), distributed)
            rgb_loss_mean = reduce_mean(rgb_loss.detach(), distributed)
            elapsed_mean = reduce_mean(torch.tensor(elapsed, device=device, dtype=torch.float32), distributed)
            running = {
                "loss": float(loss_mean.cpu()),
                "feature_loss": float(feature_loss_mean.cpu()),
                "rgb_loss": float(rgb_loss_mean.cpu()),
                "sec_per_step": float(elapsed_mean.cpu()),
                "steps_per_sec": 1.0 / max(float(elapsed_mean.cpu()), 1e-6),
            }
            if step % log_every == 0 or step == 1:
                if device.type == "cuda":
                    mem_gb = torch.tensor(torch.cuda.max_memory_allocated(device) / (1024**3), device=device, dtype=torch.float32)
                    if distributed:
                        gathered_mem = [torch.zeros_like(mem_gb) for _ in range(world_size)]
                        dist.all_gather(gathered_mem, mem_gb)
                        running["mem_gb_per_rank"] = [round(float(x.cpu()), 3) for x in gathered_mem]
                    else:
                        running["mem_gb"] = float(mem_gb.cpu())
                row = {"split": "train", "step": step, **running}
                if is_main_process(rank):
                    print(json.dumps(row, ensure_ascii=False))
                    append_jsonl(metrics_path, row)
            rank0_event = False
            if val_every > 0 and (step % val_every == 0 or step == 1):
                rank0_event = True
                if is_main_process(rank):
                    val = validate_and_visualize(ema.model, val_loader, rae, decoder, normalizer, cfg, device, out_dir / "vis", step, precision, int(cfg.training.sample_steps))
                    row = {"split": "val", "step": step, **val}
                    print(json.dumps(row, ensure_ascii=False))
                    append_jsonl(metrics_path, row)
                    plot_loss_curves(metrics_path, out_dir / "loss_curves.png")
            if infer_ckpt_every > 0 and (step % infer_ckpt_every == 0 or step == max_steps):
                rank0_event = True
                if is_main_process(rank):
                    save_inference_checkpoint(out_dir / "checkpoints_infer" / f"{step:07d}.pt", model, ema, step, cfg, normalizer)
                    save_inference_checkpoint(out_dir / "checkpoints_infer" / "last_infer.pt", model, ema, step, cfg, normalizer)
            if full_ckpt_every > 0 and (step % full_ckpt_every == 0 or step == max_steps):
                rank0_event = True
                if is_main_process(rank):
                    save_full_checkpoint(out_dir / "checkpoints_full" / f"{step:07d}.pt", model, ema, optimizer, step, cfg, normalizer)
                    save_full_checkpoint(out_dir / "checkpoints_full" / "last_full.pt", model, ema, optimizer, step, cfg, normalizer)
            if distributed and rank0_event:
                dist.barrier()
            if step >= max_steps:
                break
    total = time.perf_counter() - start_time
    report = {"max_steps": max_steps, "total_seconds": total, "avg_steps_per_sec": max_steps / max(total, 1e-6), **running}
    if is_main_process(rank):
        (out_dir / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        plot_loss_curves(metrics_path, out_dir / "loss_curves.png")
        print(json.dumps({"done": True, **report}, ensure_ascii=False))
    if distributed:
        dist.barrier()
    cleanup_distributed()


@torch.no_grad()
def rollout(args, cfg) -> None:
    add_repo_to_path()
    seed_everything(int(cfg.training.seed))
    device = torch.device(str(args.device or cfg.training.device))
    out_dir = Path(str(args.output_dir or cfg.project.output_dir)) / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    _, val_loader = build_loaders(cfg, args)
    rae, decoder = build_rae_and_decoder(cfg, device)
    model = build_denoiser(cfg).to(device)
    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    model.load_state_dict(ckpt.get("ema", ckpt.get("model", ckpt)), strict=True)
    model.eval()
    normalizer = FeatureNormalizer.from_checkpoint(ckpt if isinstance(ckpt, dict) else {}, int(cfg.da3.feature_dim), device)
    precision = str(args.precision or cfg.training.precision)
    frames = int(args.rollout_frames or cfg.rollout.frames)
    sample_steps = int(args.sample_steps or cfg.rollout.sample_steps)

    batch = next(iter(val_loader))
    images = batch["images"].to(device)
    images_01 = batch["images_01"].to(device)
    rays = batch["rays"].to(device)
    actions = make_action_condition(batch["ego_xytheta"].to(device), float(cfg.data.pose_scale))
    hist = int(cfg.data.history_frames)
    total_needed = hist + frames
    if images.shape[1] < total_needed:
        raise RuntimeError(f"Validation sample only has {images.shape[1]} frames, need {total_needed}")
    z_hist_raw = extract_da3_tokens_chunked(rae, images[:, :hist], precision, int(cfg.da3.get("encode_chunk_size", 0)))
    z_hist = normalizer.normalize(z_hist_raw)
    hist_rays = rays[:, :hist]
    future_rays = rays[:, hist : hist + frames]
    future_actions = actions[:, hist : hist + frames] if getattr(model, "uses_actions", False) else None
    g = model.encode_history(z_hist, hist_rays)
    preds = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for ti in tqdm(range(frames), desc="rollout"):
        cur_ray = future_rays[:, ti : ti + 1]
        cur_action = future_actions[:, ti : ti + 1] if future_actions is not None else None
        z_next = model.sample(
            (1, 1, len(cfg.data.camera_ids), z_hist.shape[-2], z_hist.shape[-1]),
            cur_ray,
            g,
            sample_steps,
            actions=cur_action,
        )
        preds.append(z_next)
        g = model.update_global(g, z_next, cur_ray, actions=cur_action)
    pred_z = torch.cat(preds, dim=1)
    pred_z_raw = normalizer.denormalize(pred_z)
    pred_rgb = decode_rgb(decoder, pred_z_raw, tuple(cfg.data.image_size), precision)
    gt_rgb = images_01[:, hist : hist + frames]
    save_rgb_grid(out_dir / "rollout_rgb.png", images_01[0, :hist], gt_rgb[0], pred_rgb[0])
    depth_method = str(cfg.rollout.get("depth_method", "proxy")).lower()
    if depth_method == "da3":
        pred_depth = predict_da3_depth(pred_rgb, str(cfg.da3.encoder_pretrained_path), device, precision)
        save_depth_grid(out_dir / "rollout_depth_da3.png", pred_depth[0])
    elif depth_method == "proxy":
        save_depth_proxy(out_dir / "rollout_depth_proxy.png", pred_z_raw, tuple(cfg.data.image_size))
    report = {
        "frames": frames,
        "sample_steps": sample_steps,
        "depth_method": depth_method,
        "seconds": time.perf_counter() - start,
        "rgb_l1_vs_gt": float(F.l1_loss(pred_rgb.float(), gt_rgb.float()).cpu()),
        "scene": batch["scene"][0],
        "frames_ids": batch["frames"][0, : total_needed].tolist(),
    }
    if device.type == "cuda":
        report["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
    (out_dir / "rollout_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "rollout"], default="train")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--ckpt-every", type=int, default=None)
    parser.add_argument("--infer-ckpt-every", type=int, default=None)
    parser.add_argument("--normalizer-batches", type=int, default=None)
    parser.add_argument("--rollout-frames", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.normalizer_batches is not None:
        if "normalization" not in cfg:
            cfg.normalization = {}
        cfg.normalization.num_batches = int(args.normalizer_batches)
    if args.mode == "train":
        train(args, cfg)
    else:
        if not args.ckpt:
            raise ValueError("--mode rollout requires --ckpt")
        rollout(args, cfg)


if __name__ == "__main__":
    main()
