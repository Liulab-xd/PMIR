import csv
import json
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from pmir.config import W_IMAGE, W_PHYSICS, W_POLARIZATION, W_SCALE
from pmir.data import PolarDataset
from pmir.losses import PolLoss
from pmir.models import PMIR


def train(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("pmir")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    dataset = PolarDataset(
        args.scatter_root, args.raw_root, args.image_size, augment=args.augment
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = PMIR(args.feature_channels, args.retinex).to(device)
    criterion = PolLoss().to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Device: %s | Samples: %d | Parameters: %.3f M", device, len(dataset), parameter_count / 1e6)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(vars(args))
    run_config["resolved_device"] = str(device)
    with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2)

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            component_sum = {}
            progress = tqdm(train_loader, desc=f"Train {epoch:04d}/{args.epochs}", leave=False)
            for angle_input, dop, gt_angle in progress:
                angle_input = angle_input.to(device, non_blocking=True)
                dop = dop.to(device, non_blocking=True)
                gt_angle = gt_angle.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                (angle_out, tau_hat, t_hat, beta_rgb, b_hat, b_stokes,
                 r_log, l_log, phys_conf, j_phys_raw) = model(angle_input, dop)
                loss, logs = criterion(
                    angle_out, gt_angle, angle_input, dop, tau_hat, t_hat, b_hat,
                    r_log=r_log, l_log=l_log,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                for name, value in logs.items():
                    component_sum[name] = component_sum.get(name, 0.0) + value
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    scale=f"{logs['scale']:.4f}",
                    pol=f"{logs['polarization']:.4f}",
                    tq=f"{logs['tq']:.4f}",
                    phys=f"{logs['phys']:.4f}",
                    invalid=f"{(((j_phys_raw < 0) | (j_phys_raw > 1)).float().mean()).item():.3f}",
                    pconf=f"{phys_conf.mean().item():.3f}",
                )

            scheduler.step()
            batches = len(train_loader)
            averages = {name: value / batches for name, value in component_sum.items()}
            row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **averages}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()

            weighted = {
                "image": W_IMAGE * averages["image"],
                "scale": W_SCALE * averages["scale"],
                "physics": W_PHYSICS * averages["physics"],
                "polarization": W_POLARIZATION * averages["polarization"],
            }
            logger.info(
                "Epoch %04d/%d | Loss %.5f | LR %.2e | %s",
                epoch, args.epochs, averages["total"], scheduler.get_last_lr()[0],
                " ".join(f"{name}={value:.4f}" for name, value in weighted.items()),
            )
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": averages["total"],
                "config": run_config,
            }
            latest = checkpoint_dir / "last_model.pth"
            temporary = checkpoint_dir / "last_model.pth.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(latest)
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:04d}.pth")

    logger.info("Training complete. Outputs: %s", output_dir)
