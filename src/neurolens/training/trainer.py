"""Training loop with validation, checkpointing, early stopping, AMP and resumption."""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from neurolens.data.audit import duplicate_groups_from_csv
from neurolens.data.dataset import MRIDataset
from neurolens.data.discovery import discover_dataset
from neurolens.data.splits import (
    build_manifest,
    load_manifest,
    manifest_dir,
    save_manifest,
    split_items,
)
from neurolens.data.transforms import build_eval_transform, build_train_transform
from neurolens.evaluation.metrics import compute_classification_metrics
from neurolens.models.factory import build_model, count_parameters
from neurolens.training.checkpoint import load_checkpoint, save_checkpoint
from neurolens.training.config import MONITOR_METRICS, ExperimentConfig, save_config
from neurolens.utils import (
    device_description,
    get_logger,
    read_json,
    resolve_device,
    seed_worker,
    set_seed,
    software_versions,
    utc_now,
    write_json,
)

log = get_logger("train")

HISTORY_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_balanced_accuracy",
    "val_macro_f1",
    "epoch_seconds",
]


def ensure_manifest(config: ExperimentConfig) -> Path:
    """Return the manifest for ``config``, building and saving it on first use."""
    if config.data.manifest:
        return Path(config.data.manifest)
    out_dir = manifest_dir(
        Path(config.data.splits_dir), config.data.benchmark, config.seed, config.data.val_fraction
    )
    path = out_dir / "manifest.csv"
    if path.exists():
        return path

    layout = discover_dataset(config.data.root)
    audit_dir = Path(config.data.audit_dir)
    dup_groups = None
    unreadable = None
    if (audit_dir / "audit_report.json").exists():
        report = read_json(audit_dir / "audit_report.json")
        unreadable = [item["relpath"] for item in report.get("unreadable_files", [])]
    if config.data.benchmark == "leakage_audited":
        files_csv = audit_dir / "audit_files.csv"
        if not files_csv.exists():
            raise FileNotFoundError(
                f"{files_csv} not found. Run `neurolens audit --data-root {config.data.root}` "
                "before using the leakage_audited benchmark."
            )
        dup_groups = duplicate_groups_from_csv(files_csv)
    frame, metadata = build_manifest(
        layout,
        val_fraction=config.data.val_fraction,
        seed=config.seed,
        benchmark=config.data.benchmark,
        dup_groups=dup_groups,
        unreadable=unreadable,
    )
    log.info("Created %s manifest with counts %s", config.data.benchmark, metadata["counts"])
    return save_manifest(frame, metadata, out_dir)


@dataclass
class EarlyStopping:
    patience: int
    mode: str  # "min" or "max"
    best: float | None = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        """Record ``value``; return True if it is an improvement."""
        improved = (
            self.best is None
            or (self.mode == "min" and value < self.best)
            or (self.mode == "max" and value > self.best)
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    logits: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> EpochResult:
    """One pass over ``loader``. Trains when ``optimizer`` is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, (images, labels) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                all_logits.append(logits.float().cpu())
                all_labels.append(labels.cpu())
            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (logits.argmax(1) == labels).sum().item()
            seen += batch
    if seen == 0:
        raise RuntimeError("The data loader produced no batches")
    result = EpochResult(loss=total_loss / seen, accuracy=correct / seen)
    if not training:
        result.logits = torch.cat(all_logits).numpy()
        result.labels = torch.cat(all_labels).numpy()
    return result


def make_loader(
    items: list[tuple[Path, int]],
    transform: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        MRIDataset(items, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _make_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    t = config.training
    if t.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=t.lr, momentum=0.9, weight_decay=t.weight_decay, nesterov=True
        )
    return torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)


def _make_scheduler(optimizer: torch.optim.Optimizer, config: ExperimentConfig) -> Any:
    t = config.training
    if t.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.epochs)
    if t.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=MONITOR_METRICS[t.monitor], factor=0.5, patience=2
        )
    return None


def _write_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row) + "\n")


def build_metadata(
    config: ExperimentConfig, class_names: list[str], manifest_meta: dict[str, Any]
) -> dict[str, Any]:
    return {
        "architecture": config.model.architecture,
        "class_names": class_names,
        "preprocessing": config.preprocessing.to_dict(),
        "model": {
            "dropout": config.model.dropout,
            "pretrained": config.model.pretrained,
            "freeze_backbone": config.model.freeze_backbone,
        },
        "experiment": config.to_dict(),
        "benchmark": manifest_meta.get("benchmark", config.data.benchmark),
        "split": {k: manifest_meta.get(k) for k in ("seed", "val_fraction", "counts", "procedure")},
        "software": software_versions(),
    }


def train(config: ExperimentConfig, resume: bool = False) -> Path:
    """Train according to ``config``; returns the path of the best checkpoint."""
    config.validate()
    set_seed(config.seed, config.training.deterministic)
    device = resolve_device(config.training.device)
    use_amp = config.training.amp and device.type == "cuda"
    run_dir = config.run_dir
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if run_dir.exists() and last_path.exists() and not resume:
        raise FileExistsError(
            f"Run '{config.experiment_name}' already exists. Use --resume to continue it or "
            "choose another experiment_name."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ensure_manifest(config)
    frame, manifest_meta = load_manifest(manifest_path, Path(config.data.root))
    class_names = list(manifest_meta.get("class_names") or sorted(frame["class_name"].unique()))
    shutil.copy(manifest_path, run_dir / "manifest.csv")
    if manifest_path.with_name("manifest_meta.json").exists():
        shutil.copy(manifest_path.with_name("manifest_meta.json"), run_dir / "manifest_meta.json")
    save_config(config, run_dir / "config.yaml")

    t = config.training
    train_loader = make_loader(
        split_items(frame, "train"),
        build_train_transform(config.preprocessing, config.augmentation),
        t.batch_size,
        True,
        config.data.num_workers,
        config.seed,
        device,
    )
    val_loader = make_loader(
        split_items(frame, "val"),
        build_eval_transform(config.preprocessing),
        t.batch_size,
        False,
        config.data.num_workers,
        config.seed,
        device,
    )

    model = build_model(
        config.model.architecture,
        len(class_names),
        config.model.pretrained,
        config.model.dropout,
        config.model.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopping(t.early_stopping_patience, MONITOR_METRICS[t.monitor])
    metadata = build_metadata(config, class_names, manifest_meta)
    history: list[dict[str, Any]] = []
    start_epoch = 1

    if resume and last_path.exists():
        state = load_checkpoint(last_path)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        if state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        stopper.best = state["early_stopping"]["best"]
        stopper.bad_epochs = state["early_stopping"]["bad_epochs"]
        history = state["history"]
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["rng"]["torch"])
        log.info("Resumed '%s' from epoch %d", config.experiment_name, state["epoch"])
        if stopper.should_stop or start_epoch > t.epochs:
            log.info("Run already finished; nothing to resume")
            return best_path

    params = count_parameters(model)
    log.info(
        "Training %s on %s | train=%d val=%d | params=%s | amp=%s",
        config.model.architecture,
        device_description(device),
        len(train_loader.dataset),
        len(val_loader.dataset),
        f"{params['total']:,}",
        use_amp,
    )

    for epoch in range(start_epoch, t.epochs + 1):
        started = time.perf_counter()
        train_res = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp, t.max_train_batches
        )
        val_res = run_epoch(
            model, val_loader, criterion, device, use_amp=use_amp, max_batches=t.max_eval_batches
        )
        probs = torch.softmax(torch.from_numpy(val_res.logits), dim=1).numpy()
        val_metrics = compute_classification_metrics(val_res.labels, probs, class_names)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_res.loss, 6),
            "train_accuracy": round(train_res.accuracy, 6),
            "val_loss": round(val_res.loss, 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_balanced_accuracy": round(val_metrics["balanced_accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "epoch_seconds": round(time.perf_counter() - started, 2),
        }
        history.append(row)
        _write_history(run_dir, history)
        monitored = row[t.monitor]
        improved = stopper.step(monitored)
        log.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f val_f1=%.4f %s",
            epoch,
            t.epochs,
            row["train_loss"],
            row["val_loss"],
            row["val_accuracy"],
            row["val_macro_f1"],
            "*best*" if improved else "",
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitored)
            else:
                scheduler.step()

        if improved:
            best_meta = {
                **metadata,
                "epoch": epoch,
                "monitor": {"metric": t.monitor, "value": monitored},
                "val_metrics": {k: row[k] for k in HISTORY_FIELDS if k.startswith("val_")},
                "model_version": f"{config.experiment_name}-e{epoch:03d}",
                "created_at": utc_now(),
            }
            save_checkpoint(best_path, {"model_state": model.state_dict(), "metadata": best_meta})
            # Sidecar so registries can list models without loading weights.
            write_json(run_dir / "best_meta.json", best_meta)
        save_checkpoint(
            last_path,
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "early_stopping": {"best": stopper.best, "bad_epochs": stopper.bad_epochs},
                "history": history,
                "epoch": epoch,
                "rng": {"torch": torch.get_rng_state()},
                "metadata": metadata,
            },
        )
        if stopper.should_stop:
            log.info("Early stopping after %d epochs without improvement", stopper.bad_epochs)
            break

    write_json(
        run_dir / "training_summary.json",
        {
            "experiment_name": config.experiment_name,
            "architecture": config.model.architecture,
            "device": device_description(device),
            "epochs_completed": len(history),
            "best_epoch": max(
                history, key=lambda r: r[t.monitor] * (1 if stopper.mode == "max" else -1)
            )["epoch"],
            "monitor": t.monitor,
            "best_value": stopper.best,
            "parameters": params,
            "finished_at": utc_now(),
        },
    )
    return best_path
