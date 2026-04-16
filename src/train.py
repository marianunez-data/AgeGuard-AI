# ══════════════════════════════════════════════════════════════════
#  src/train.py
#  Training loop for AgeGuard AI age estimation model.
#
#  Features:
#    - Huber Loss: robust to UTKFace mislabels
#    - Resume training: recovers from interruptions
#    - Dual checkpointing: best_model.pth + last_checkpoint.pth
#    - Training curves: saved as PNG + JSON
#    - Mixed precision (AMP): faster training on GPU
# ══════════════════════════════════════════════════════════════════
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one epoch. Returns avg loss and MAE."""
    model.train()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0

    for imgs, ages in loader:
        imgs = imgs.to(device)
        ages = ages.to(device).unsqueeze(1)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            preds = model(imgs)
            loss = criterion(preds, ages)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        running_mae += (preds - ages).abs().sum().item()
        n_samples += batch_size

    return running_loss / n_samples, running_mae / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    """Evaluate on val/test set. Returns avg loss and MAE."""
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    n_samples = 0

    for imgs, ages in loader:
        imgs = imgs.to(device)
        ages = ages.to(device).unsqueeze(1)

        with autocast(enabled=use_amp):
            preds = model(imgs)
            loss = criterion(preds, ages)

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        running_mae += (preds - ages).abs().sum().item()
        n_samples += batch_size

    return running_loss / n_samples, running_mae / n_samples


def save_training_curves(history, save_dir, best_epoch):
    """Generate and save training curves as PNG."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("AgeGuard AI — Training Curves", fontsize=14, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0].axvline(
        best_epoch,
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"Best (epoch {best_epoch})",
    )
    axes[0].set_title("Huber Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE curves
    axes[1].plot(epochs, history["train_mae"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_mae"], "r-", label="Val", linewidth=2)
    axes[1].axhline(
        5.0, color="green", linestyle="--", alpha=0.5, label="Target MAE (5.0)"
    )
    axes[1].axvline(best_epoch, color="green", linestyle=":", alpha=0.5)
    axes[1].set_title("Mean Absolute Error (years)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning rate
    axes[2].plot(epochs, history["lr"], "g-", linewidth=2)
    axes[2].set_title("Learning Rate (CosineAnnealing)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"  Curves saved → {save_path}")


def train(cfg, model, loaders, device):
    """
    Full training loop with resume, early stopping and checkpointing.

    Saves two checkpoints:
      - best_model.pth: best validation MAE (for production)
      - last_checkpoint.pth: every epoch (for crash recovery)
    """
    model = model.to(device)

    # Loss: Huber is robust to label noise
    criterion = nn.HuberLoss(delta=1.0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs, eta_min=1e-6
    )

    # Mixed precision
    scaler = GradScaler(enabled=cfg.training.use_amp)

    # Tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "lr": [],
    }

    best_val_mae = float("inf")
    patience_counter = 0
    best_epoch = 0
    start_epoch = 1

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    best_path = cfg.models_dir / "best_model.pth"
    last_path = cfg.models_dir / "last_checkpoint.pth"

    # ── Resume from checkpoint if exists ──────────────────────────
    if last_path.exists():
        print("  Resuming from last checkpoint...")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_mae = checkpoint["best_val_mae"]
        best_epoch = checkpoint["best_epoch"]
        history = checkpoint["history"]
        patience_counter = checkpoint["patience_counter"]
        print(
            f'  Resumed from epoch {checkpoint["epoch"]} '
            f"(best MAE: {best_val_mae:.2f} at epoch {best_epoch})"
        )

    print("\nTraining Configuration")
    print("=" * 55)
    print(f"  Device          : {device}")
    print(
        f'  GPU             : {torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A"}'
    )
    print(f"  Epochs          : {start_epoch} → {cfg.training.epochs}")
    print(f"  Batch size      : {cfg.training.batch_size}")
    print(f"  Learning rate   : {cfg.training.lr}")
    print(f"  Weight decay    : {cfg.training.weight_decay}")
    print(f"  AMP             : {cfg.training.use_amp}")
    print(f"  Patience        : {cfg.training.patience}")
    print("  Loss            : HuberLoss (delta=1.0)")
    print("  Optimizer       : AdamW")
    print("  Scheduler       : CosineAnnealingLR")
    print()

    start_time = time.time()

    for epoch in range(start_epoch, cfg.training.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_mae = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scaler,
            device,
            cfg.training.use_amp,
        )

        # Validate
        val_loss, val_mae = evaluate(
            model, loaders["val"], criterion, device, cfg.training.use_amp
        )

        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Track history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        epoch_time = time.time() - epoch_start

        # Check improvement
        improved = ""
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mae": val_mae,
                    "val_loss": val_loss,
                },
                best_path,
            )
            improved = " ★ best"
        else:
            patience_counter += 1

        # Save last checkpoint (crash recovery)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "history": history,
            },
            last_path,
        )

        print(
            f"  Epoch {epoch:2d}/{cfg.training.epochs} "
            f"| train_loss: {train_loss:.4f} "
            f"| val_loss: {val_loss:.4f} "
            f"| train_MAE: {train_mae:.2f} "
            f"| val_MAE: {val_mae:.2f} "
            f"| lr: {current_lr:.1e} "
            f"| {epoch_time:.0f}s{improved}"
        )

        # Early stopping
        if patience_counter >= cfg.training.patience:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for {cfg.training.patience} epochs)"
            )
            break

    total_time = time.time() - start_time

    # ── Training curves ───────────────────────────────────────────
    reports_dir = cfg.base_dir / "reports" / "training"
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_training_curves(history, reports_dir, best_epoch)

    # ── Training summary ──────────────────────────────────────────
    summary = {
        "run_date": datetime.now().isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "total_epochs_run": epoch - start_epoch + 1,
        "best_epoch": best_epoch,
        "best_val_mae": round(best_val_mae, 4),
        "final_train_mae": round(history["train_mae"][-1], 4),
        "final_val_mae": round(history["val_mae"][-1], 4),
        "mae_target": cfg.business.mae_target,
        "gap_to_target": round(best_val_mae - cfg.business.mae_target, 4),
        "total_time_minutes": round(total_time / 60, 1),
        "config": {
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay,
            "patience": cfg.training.patience,
            "use_amp": cfg.training.use_amp,
            "loss": "HuberLoss_delta1.0",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        "history": history,
    }

    print("\nTraining Complete")
    print("=" * 55)
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Best val MAE    : {best_val_mae:.2f} years")
    print(f'  Final train MAE : {history["train_mae"][-1]:.2f} years')
    print(f'  Final val MAE   : {history["val_mae"][-1]:.2f} years')
    print(f"  Target MAE      : {cfg.business.mae_target} years")
    print(f"  Total time      : {total_time/60:.1f} minutes")
    print(f"  Best checkpoint : {best_path}")
    print(f"  Last checkpoint : {last_path}")

    return summary
