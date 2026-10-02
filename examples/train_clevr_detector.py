"""Fine-tune a one-class Faster R-CNN detector on approximate CLEVR boxes."""

import argparse
from pathlib import Path
import random

from spinls.detection_training import (
    CLEVRDetectionDataset,
    build_clevr_detector,
    detection_collate,
    detector_checkpoint,
    evaluate_detector,
)


def atomic_save(payload, path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--min-size", type=int, default=320)
    parser.add_argument("--max-size", type=int, default=480)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--limit-train-scenes", type=int)
    parser.add_argument("--limit-val-scenes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers must be nonnegative")

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    train_data = CLEVRDetectionDataset(
        args.data_dir, "train", limit=args.limit_train_scenes
    )
    val_data = CLEVRDetectionDataset(args.data_dir, "val", limit=args.limit_val_scenes)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": detection_collate,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume.get("schema_version") != 1:
            parser.error("unsupported resume checkpoint")
        if (resume["min_size"], resume["max_size"]) != (args.min_size, args.max_size):
            parser.error("resume checkpoint image sizes differ from command-line values")
    model = build_clevr_detector(
        pretrained=resume is None, min_size=args.min_size, max_size=args.max_size
    ).to(args.device)
    if resume:
        model.load_state_dict(resume["model_state_dict"])

    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    start_epoch = 1
    if resume:
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        start_epoch = resume["epoch"] + 1

    last_path = args.output.with_name(args.output.stem + ".last" + args.output.suffix)
    best_f1 = -1.0
    if args.output.exists():
        best_f1 = torch.load(
            args.output, map_location="cpu", weights_only=True
        ).get("validation", {}).get("f1", -1.0)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        objects = 0
        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(args.device, non_blocking=True) for image in images]
            targets = [
                {name: value.to(args.device, non_blocking=True) for name, value in target.items()}
                for target in targets
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.device.startswith("cuda")):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_objects = sum(len(target["boxes"]) for target in targets)
            running_loss += float(loss.detach()) * batch_objects
            objects += batch_objects
            if batch_index % 100 == 0:
                print(
                    f"epoch {epoch} batch {batch_index}/{len(train_loader)} "
                    f"loss={running_loss / objects:.4f}",
                    flush=True,
                )
        scheduler.step()
        metrics = evaluate_detector(
            model,
            val_loader,
            device=args.device,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
        )
        metrics["train_loss"] = running_loss / objects
        print(
            f"epoch {epoch}: loss={metrics['train_loss']:.4f}, "
            f"precision={metrics['precision']:.2%}, recall={metrics['recall']:.2%}, "
            f"F1={metrics['f1']:.2%}, IoU={metrics['mean_matched_iou']:.3f}, "
            f"count_MAE={metrics['mean_absolute_count_error']:.3f}",
            flush=True,
        )
        last_checkpoint = detector_checkpoint(
            model,
            epoch=epoch,
            metrics=metrics,
            min_size=args.min_size,
            max_size=args.max_size,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        atomic_save(last_checkpoint, last_path)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_checkpoint = detector_checkpoint(
                model,
                epoch=epoch,
                metrics=metrics,
                min_size=args.min_size,
                max_size=args.max_size,
            )
            atomic_save(best_checkpoint, args.output)
            print(f"saved new best detector to {args.output}", flush=True)


if __name__ == "__main__":
    main()
