# train.py

from __future__ import annotations

import argparse
import random
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import wandb

from datasets import build_dataset
from models import build_model
from losses import build_loss
from metrics import METRICS


# -------------------------------------------------------------------------
# Config / utilities
# -------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize_slices(moving: torch.Tensor, fixed: torch.Tensor, warped: torch.Tensor):
    """
    Create W&B image list with mid slices from moving/fixed/warped volumes.

    moving, fixed, warped: (B, 1, D, H, W)
    """
    mid = moving.shape[2] // 2  # axial slice index
    images = []
    for name, vol in [("moving", moving), ("fixed", fixed), ("warped", warped)]:
        # Use first sample in batch
        slice_img = vol[0, 0, mid].detach().cpu().numpy()
        images.append(wandb.Image(slice_img, caption=name))
    return images


# -------------------------------------------------------------------------
# Training / evaluation loops
# -------------------------------------------------------------------------

def train_one_epoch(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        device: torch.device,
        scaler: GradScaler,
        use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        # 基本输入
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)

        # 可选：DeepReg synthetic DVF 监督
        gt_dvf = batch.get("dvf", None)
        if gt_dvf is not None:
            gt_dvf = gt_dvf.to(device, non_blocking=True)

        # NaN/Inf 检查
        if torch.isnan(moving).any() or torch.isinf(moving).any():
            print("Warning: NaN/Inf detected in moving image. Skipping batch.")
            continue
        if torch.isnan(fixed).any() or torch.isinf(fixed).any():
            print("Warning: NaN/Inf detected in fixed image. Skipping batch.")
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            # 模型预测：同时拿 warped 和 pred_dvf (ddf)
            warped, ddf = model(moving, fixed)

            # 输出 NaN/Inf 检查
            if torch.isnan(warped).any() or torch.isinf(warped).any():
                print("Warning: NaN/Inf detected in warped output. Skipping batch.")
                continue
            if torch.isnan(ddf).any() or torch.isinf(ddf).any():
                print("Warning: NaN/Inf detected in ddf output. Skipping batch.")
                continue

            # **关键：如果有 gt_dvf，就用 DVF 监督的 loss**
            if gt_dvf is not None:
                loss = loss_fn(warped, fixed, ddf, gt_dvf)
            else:
                # 兼容原来的 LNCC-only 配置
                loss = loss_fn(warped, fixed)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = moving.shape[0]
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / max(n_samples, 1)


"""
def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)

        # Check inputs for NaN/Inf
        if torch.isnan(moving).any() or torch.isinf(moving).any():
            print(f"Warning: NaN/Inf detected in moving image. Skipping batch.")
            continue
        if torch.isnan(fixed).any() or torch.isinf(fixed).any():
            print(f"Warning: NaN/Inf detected in fixed image. Skipping batch.")
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            warped, ddf = model(moving, fixed)

            # Check model outputs for NaN/Inf
            if torch.isnan(warped).any() or torch.isinf(warped).any():
                print(f"Warning: NaN/Inf detected in warped output. Skipping batch.")
                continue
            if torch.isnan(ddf).any() or torch.isinf(ddf).any():
                print(f"Warning: NaN/Inf detected in ddf output. Skipping batch.")
                continue

            loss = loss_fn(warped, fixed)

        # Check for NaN/inf loss
        # if torch.isnan(loss) or torch.isinf(loss):
        #    print(f"Warning: NaN/Inf loss detected. Loss value: {loss.item()}. Skipping batch.")
        #    continue

        if use_amp:
            scaler.scale(loss).backward()
            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = moving.shape[0]
        running_loss += loss.item() * bs
        n_samples += bs

    return running_loss / max(n_samples, 1)
"""


def evaluate(
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        use_amp: bool,
):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    metric_totals = {k: 0.0 for k in METRICS}
    visuals = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            moving = batch["moving"].to(device, non_blocking=True)
            fixed = batch["fixed"].to(device, non_blocking=True)
            gt_dvf = batch.get("dvf", None)
            if gt_dvf is not None:
                gt_dvf = gt_dvf.to(device, non_blocking=True)

            # NaN/Inf 检查
            if torch.isnan(moving).any() or torch.isinf(moving).any():
                print("Warning: NaN/Inf detected in moving image (eval). Skipping batch.")
                continue
            if torch.isnan(fixed).any() or torch.isinf(fixed).any():
                print("Warning: NaN/Inf detected in fixed image (eval). Skipping batch.")
                continue

            with autocast("cuda", enabled=use_amp):
                warped, ddf = model(moving, fixed)

                if torch.isnan(warped).any() or torch.isinf(warped).any():
                    print("Warning: NaN/Inf detected in warped output (eval). Skipping batch.")
                    continue
                if torch.isnan(ddf).any() or torch.isinf(ddf).any():
                    print("Warning: NaN/Inf detected in ddf output (eval). Skipping batch.")
                    continue

                if gt_dvf is not None:
                    loss = loss_fn(warped, fixed, ddf, gt_dvf)
                else:
                    loss = loss_fn(warped, fixed)

            bs = moving.shape[0]
            running_loss += loss.item() * bs
            n_samples += bs

            # 评估指标
            for name, fn in METRICS.items():
                if name == "grad_l2":
                    # 对预测 DVF 求梯度范数
                    metric_totals[name] += fn(ddf) * bs
                elif name == "epe" and gt_dvf is not None:
                    # EPE 专门比较 pred_dvf vs gt_dvf
                    metric_totals[name] += fn(ddf, gt_dvf) * bs
                else:
                    # 像 mse / mae / ncc 一样走图像空间
                    metric_totals[name] += fn(warped, fixed) * bs

            # 只存第一 batch 的可视化
            if batch_idx == 0:
                visuals = visualize_slices(moving, fixed, warped)

    avg_loss = running_loss / max(n_samples, 1)
    avg_metrics = {
        name: metric_totals[name] / max(n_samples, 1) for name in METRICS
    }
    return avg_loss, avg_metrics, visuals


"""
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    metric_totals = {k: 0.0 for k in METRICS}
    visuals = None

    for batch_idx, batch in enumerate(dataloader):
        moving = batch["moving"].to(device, non_blocking=True)
        fixed = batch["fixed"].to(device, non_blocking=True)

        # Check inputs for NaN/Inf
        if torch.isnan(moving).any() or torch.isinf(moving).any():
            print(f"Warning: NaN/Inf detected in moving image (eval). Skipping batch.")
            continue
        if torch.isnan(fixed).any() or torch.isinf(fixed).any():
            print(f"Warning: NaN/Inf detected in fixed image (eval). Skipping batch.")
            continue

        with autocast("cuda", enabled=use_amp):
            warped, ddf = model(moving, fixed)

            # Check model outputs for NaN/Inf
            if torch.isnan(warped).any() or torch.isinf(warped).any():
                print(f"Warning: NaN/Inf detected in warped output (eval). Skipping batch.")
                continue
            if torch.isnan(ddf).any() or torch.isinf(ddf).any():
                print(f"Warning: NaN/Inf detected in ddf output (eval). Skipping batch.")
                continue

            loss = loss_fn(warped, fixed)

        # Check for NaN/inf loss
        # if torch.isnan(loss) or torch.isinf(loss):
        #    print(f"Warning: NaN/Inf loss detected in evaluation. Loss value: {loss.item()}. Skipping batch.")
        #    continue

        bs = moving.shape[0]
        running_loss += loss.item() * bs
        n_samples += bs

        # Compute metrics
        for name, fn in METRICS.items():
            if name == "grad_l2":
                metric_totals[name] += fn(ddf) * bs
            else:
                metric_totals[name] += fn(warped, fixed) * bs

        # Save first batch for visualization
        if batch_idx == 0:
            visuals = visualize_slices(moving, fixed, warped)

    avg_loss = running_loss / max(n_samples, 1)
    avg_metrics = {
        name: metric_totals[name] / max(n_samples, 1) for name in METRICS
    }
    return avg_loss, avg_metrics, visuals
"""


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment1.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- 现在 cfg 已经存在，才能使用 cfg["wandb"] ----
    if cfg["wandb"]["enabled"]:
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["run_name"],
            config=cfg,
            mode="offline" if cfg["wandb"]["offline"] else "online",
        )

    device_str = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")

    set_seed(cfg["training"]["seed"])

    # Datasets / loaders
    train_ds = build_dataset(cfg["train_dataset"], split="train")
    val_ds = build_dataset(cfg["val_dataset"], split="val")

    print("---- Dataset sanity check ----")
    item = val_ds[0]
    print("moving:", float(item["moving"].min()), float(item["moving"].max()))
    print("fixed:", float(item["fixed"].min()), float(item["fixed"].max()))
    print("any NaN:", torch.isnan(item["moving"]).any().item(), torch.isnan(item["fixed"]).any().item())
    print("any Inf:", torch.isinf(item["moving"]).any().item(), torch.isinf(item["fixed"]).any().item())
    print("--------------------------------")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # Model / loss / optimizer
    model = build_model(cfg["model"], image_size=cfg["image_size"]).to(device)
    loss_fn = build_loss(cfg["loss"]).to(device)

    optimizer_name = cfg["optimizer"].get("name", "Adam").lower()
    lr = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"].get("weight_decay", 0.0))

    # Safety check: if learning rate is too high, reduce it
    if lr > 1e-3:
        print(f"Warning: Learning rate {lr} is very high. Consider using a lower value (e.g., 1e-4 or 1e-5).")
    # If NaN issues persist, try even lower: 1e-5 or 1e-6

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")

    scaler = GradScaler(enabled=use_amp)

    epochs = cfg["training"]["epochs"]
    val_every = cfg["training"]["val_every"]

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    last_val_loss = float("nan")
    last_metrics = None  # 记录最近一次验证的所有指标

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        log_dict = {"train/loss": train_loss, "epoch": epoch}

        if val_every > 0 and (epoch % val_every == 0):
            val_loss, metrics, visuals = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=val_loader,
                device=device,
                use_amp=use_amp,
            )
            last_val_loss = val_loss
            last_metrics = metrics  # <-- 记录最近一次 val 的所有 metric

            log_dict["val/loss"] = val_loss
            for name, value in metrics.items():
                log_dict[f"val/{name}"] = value

            if visuals is not None:
                wandb.log({"val/slices": visuals, "epoch": epoch})

        wandb.log(log_dict)

        # -------- 在终端打印更详细的 metric 信息（含 EPE）--------
        metric_str = ""
        if last_metrics is not None:
            if "ncc" in last_metrics:
                metric_str += f", val_ncc = {last_metrics['ncc']:.4f}"
            if "epe" in last_metrics:
                metric_str += f", val_epe = {last_metrics['epe']:.4f}"
            if "grad_l2" in last_metrics:
                metric_str += f", val_grad_l2 = {last_metrics['grad_l2']:.4f}"

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss = {train_loss:.4f}, "
            f"val_loss = {last_val_loss:.4f}"
            f"{metric_str}"
        )
        # 循环结束后再统一 finish
    if cfg["wandb"]["enabled"]:
        wandb.finish()


"""
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        log_dict = {"train/loss": train_loss, "epoch": epoch}

        if val_every > 0 and (epoch % val_every == 0):
            val_loss, metrics, visuals = evaluate(
                model=model,
                loss_fn=loss_fn,
                dataloader=val_loader,
                device=device,
                use_amp=use_amp,
            )
            last_val_loss = val_loss
            log_dict["val/loss"] = val_loss
            for name, value in metrics.items():
                log_dict[f"val/{name}"] = value

            if visuals is not None:
                wandb.log({"val/slices": visuals, "epoch": epoch})

        wandb.log(log_dict)

        print(
            f"[Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss = {train_loss:.4f}, "
            f"val_loss = {last_val_loss:.4f}"
        )

         wandb.finish()
"""

if __name__ == "__main__":
    main()
