from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from timm.models._helpers import load_checkpoint
from tqdm import tqdm

from .data import build_dataset_bundle
from .dynamic import DynamicDAFManager, SensitivityAnalyzer, select_spt_like, wrap_vit_lora


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Dynamic Adaptive Fine-tuning for ViT")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/cifar100")
    parser.add_argument("--model", default="vit_base_patch16_224.augreg_in21k")
    parser.add_argument("--method", choices=("daf", "head-only"), default="daf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--split-qkv-lora", action="store_true")
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--category-ratios", default="")
    parser.add_argument("--selection-mode", choices=("category", "global"), default="category")
    parser.add_argument("--spt-sparse", action="store_true")
    parser.add_argument("--spt-alpha", type=float, default=10.0)
    parser.add_argument("--spt-beta", type=float, default=5.0)
    parser.add_argument("--update-interval", type=int, default=10)
    parser.add_argument("--sensitivity-batches", type=int, default=16)
    parser.add_argument(
        "--sensitivity-score-mode",
        choices=("grad_weight_abs", "grad_square"),
        default="grad_weight_abs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--strong-aug", action="store_true")
    parser.add_argument(
        "--train-resize-mode",
        choices=("random-resized-crop", "random-resized-crop-nohflip", "direct-resize", "direct-resize-hflip"),
        default="random-resized-crop",
    )
    parser.add_argument(
        "--eval-resize-mode",
        choices=("resize-center-crop", "direct-resize"),
        default="resize-center-crop",
    )
    parser.add_argument("--norm-mean", default="0.485,0.456,0.406")
    parser.add_argument("--norm-std", default="0.229,0.224,0.225")
    parser.add_argument("--vtab-train-list", default="train800.txt")
    parser.add_argument("--vtab-val-list", default="val200.txt")
    parser.add_argument("--vtab-test-list", default="test.txt")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--test-on-best-val-only", action="store_true")
    return parser.parse_args(argv)


def parse_float_tuple(value: str) -> tuple[float, float, float]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parsed) != 3:
        raise ValueError(f"expected three comma-separated floats, got: {value}")
    return parsed


def parse_category_ratios(value: str) -> dict[str, float] | None:
    if not value:
        return None
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parsed) != 3:
        raise ValueError(f"expected matrix,vector,special ratios, got: {value}")
    return {"matrix": parsed[0], "vector": parsed[1], "special": parsed[2]}


def can_reuse_eval_stats(args: argparse.Namespace) -> bool:
    return args.vtab_val_list == args.vtab_test_list


def should_update_best(
    current_val: float,
    best_val: float,
    current_loss: float | None = None,
    best_loss: float | None = None,
) -> bool:
    if current_val > best_val:
        return True
    if current_val < best_val:
        return False
    if current_loss is None or best_loss is None:
        return True
    return current_loss <= best_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    image_size: int,
    checkpoint_path: str | Path | None = None,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    checkpoint = str(checkpoint_path) if checkpoint_path else ""
    model = timm.create_model(
        model_name,
        pretrained=pretrained and not checkpoint,
        num_classes=num_classes,
        img_size=image_size,
        drop_path_rate=drop_path_rate,
    )
    if checkpoint:
        load_checkpoint(model, checkpoint, strict=False)
    return model


def set_head_only_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    head = getattr(model, "head", None)
    if head is None:
        raise RuntimeError("head-only training requires the model to expose a head module")
    for param in head.parameters():
        param.requires_grad = True


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("no trainable parameters selected")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)


def adjust_learning_rate(optimizer: torch.optim.Optimizer, progress_epoch: float, args: argparse.Namespace) -> float:
    if progress_epoch < args.warmup_epochs:
        lr = args.lr * progress_epoch / max(1, args.warmup_epochs)
    else:
        progress = (progress_epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, (samples, targets) in enumerate(progress):
        lr = adjust_learning_rate(optimizer, epoch + step / max(1, len(loader)), args)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(samples)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += batch_size
        progress.set_postfix(loss=total_loss / max(1, total_count), lr=f"{lr:.2e}")
    return {"loss": total_loss / total_count, "acc1": total_correct / total_count * 100.0}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, name: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    for samples, targets in tqdm(loader, desc=name, leave=False):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(samples)
        loss = criterion(logits, targets)
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += batch_size
    return {"loss": total_loss / total_count, "acc1": total_correct / total_count * 100.0}


def save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    bundle = build_dataset_bundle(
        dataset=args.dataset,
        root=args.data_root,
        seed=args.seed,
        image_size=args.image_size,
        download=not args.no_download,
        strong_aug=args.strong_aug,
        train_resize_mode=args.train_resize_mode,
        eval_resize_mode=args.eval_resize_mode,
        train_size=args.train_size,
        val_size=args.val_size,
        mean=parse_float_tuple(args.norm_mean),
        std=parse_float_tuple(args.norm_std),
        vtab_train_list=args.vtab_train_list,
        vtab_val_list=args.vtab_val_list,
        vtab_test_list=args.vtab_test_list,
    )
    train_loader = DataLoader(
        bundle.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    sensitivity_loader = DataLoader(
        bundle.train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(bundle.val, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(bundle.test, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
    reuse_eval_stats = can_reuse_eval_stats(args)

    model = create_model(
        args.model,
        bundle.num_classes,
        pretrained=not args.no_pretrained,
        image_size=args.image_size,
        checkpoint_path=args.checkpoint or None,
        drop_path_rate=args.drop_path_rate,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    analyzer = None
    manager = None
    if args.method == "daf":
        wrapped = wrap_vit_lora(
            model,
            rank=args.rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            split_qkv=args.split_qkv_lora,
        )
        if not wrapped:
            raise RuntimeError(f"no LoRA target modules found in model {args.model}")
        analyzer = SensitivityAnalyzer(
            batch_num=args.sensitivity_batches,
            rank=args.rank,
            score_mode=args.sensitivity_score_mode,
        )
        manager = DynamicDAFManager(
            budget_ratio=args.budget,
            update_interval=args.update_interval,
            category_ratios=parse_category_ratios(args.category_ratios),
            selection_mode=args.selection_mode,
        )
        if args.spt_sparse:
            initial_scores = analyzer.analyze_scores(model, sensitivity_loader, criterion, device)
            initial_selection = select_spt_like(
                model,
                initial_scores,
                budget_million=args.budget,
                rank=args.rank,
                alpha=args.spt_alpha,
                beta=args.spt_beta,
            )
            selection = manager.initialize_selection(model, initial_selection)
        else:
            initial_candidates = analyzer.analyze(model, sensitivity_loader, criterion, device)
            selection = manager.initialize(model, initial_candidates)
    else:
        set_head_only_trainable(model)
        selection = None
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    best_val = -1.0
    best_val_loss = float("inf")
    best_test = -1.0
    metrics_path = output_dir / "metrics.jsonl"
    start = time.time()
    for epoch in range(args.epochs):
        if args.method == "daf" and epoch > 0 and epoch % args.update_interval == 0:
            assert analyzer is not None
            assert manager is not None
            if args.spt_sparse:
                scores = analyzer.analyze_scores(model, sensitivity_loader, criterion, device)
                next_selection = select_spt_like(
                    model,
                    scores,
                    budget_million=args.budget,
                    rank=args.rank,
                    alpha=args.spt_alpha,
                    beta=args.spt_beta,
                )
                selection = manager.maybe_update_selection(epoch, model, next_selection) or selection
            else:
                candidates = analyzer.analyze(model, sensitivity_loader, criterion, device)
                selection = manager.maybe_update(epoch, model, candidates) or selection
            optimizer = build_optimizer(model, args.lr, args.weight_decay)

        train_stats = run_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        should_eval = epoch % args.eval_interval == 0 or epoch >= args.epochs - 10 or epoch == args.epochs - 1
        if should_eval:
            val_stats = evaluate(model, val_loader, criterion, device, "val")
            is_best = should_update_best(val_stats["acc1"], best_val, val_stats["loss"], best_val_loss)
            should_test = not args.test_on_best_val_only or is_best or epoch == args.epochs - 1
            if should_test and reuse_eval_stats:
                test_stats = dict(val_stats)
            elif should_test:
                test_stats = evaluate(model, test_loader, criterion, device, "test")
            else:
                test_stats = {"loss": float("nan"), "acc1": float("nan")}
            if is_best:
                best_val = val_stats["acc1"]
                best_val_loss = val_stats["loss"]
                best_test = test_stats["acc1"]
                save_checkpoint(output_dir / "best.pt", model, epoch, {"val": val_stats, "test": test_stats})
        else:
            val_stats = {"loss": float("nan"), "acc1": float("nan")}
            test_stats = {"loss": float("nan"), "acc1": float("nan")}
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch + 1}.pt", model, epoch, {"val": val_stats, "test": test_stats})

        row = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
            "best_val_acc1": best_val,
            "best_val_loss": best_val_loss,
            "best_test_acc1_at_best_val": best_test,
            "selected_cost": selection.total_cost if selection is not None else 0,
            "selected_budget": selection.budget if selection is not None else 0,
            "selected_counts": {key: len(value) for key, value in selection.names_by_category.items()}
            if selection is not None
            else {},
            "elapsed_sec": time.time() - start,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
