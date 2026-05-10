#!/usr/bin/env python
"""Quick dataset sanity check before launching GLD stage-1 training."""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="reproduce_gld_on_waymo/DA3_stage1_mae_waymo_672x448.yaml")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    from cut3r_data import get_data_loader

    cfg = OmegaConf.load(args.config)
    dataset_expr = cfg.train_dataset if args.split == "train" else cfg.test_dataset
    loader = get_data_loader(
        dataset_expr,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_mem=False,
        world_size=1,
        rank=0,
        fixed_length=True,
    )
    loader.dataset.set_epoch(0)
    if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(0)

    image_dict = next(iter(loader))
    images = torch.stack([d["img"] for d in image_dict], dim=1)
    labels = [d.get("label", [""])[0] for d in image_dict]
    print(f"split={args.split}")
    print(f"num_dataset_items={len(loader.dataset)}")
    print(f"batch_images_shape={tuple(images.shape)}  # (B,V,C,H,W), ImageNet-normalized")
    print("sample_labels:")
    for label in labels:
        print(f"  {label}")


if __name__ == "__main__":
    main()
