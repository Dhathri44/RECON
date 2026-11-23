"""
RECON BraTS U-Net inpainting training.

Designed for Google Colab + CUDA (e.g. Tesla T4).
Keeps the existing ResidualUNet(in_channels=2, base=32) interface and
saves checkpoints compatible with src.unet_inpainting.load_inpainting_model.

Important:
- Patient-level train/validation/test split.
- One NIfTI volume is decompressed once per patient batch, not once per slice.
- Uses non-empty axial slices, 240x240 inputs, rectangle + brush masks.
- AMP on CUDA, progress bars, early stopping, best + last checkpoints.
- Test metrics are computed only after selecting the best validation checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import List

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm

from src.unet_inpainting import ResidualUNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_flair_files(root: Path):
    files = sorted(root.rglob("*_flair.nii.gz"))
    if not files:
        raise RuntimeError(f"No *_flair.nii.gz files found under: {root}")
    return files


class PatientBatchSampler(Sampler[List[int]]):
    """
    Yields all samples for one patient together.
    This lets the dataset keep only one decompressed NIfTI volume in memory.
    Each yielded batch contains samples_per_patient indices.
    """
    def __init__(self, number_of_patients: int, samples_per_patient: int,
                 shuffle_patients: bool, seed: int):
        self.number_of_patients = number_of_patients
        self.samples_per_patient = samples_per_patient
        self.shuffle_patients = shuffle_patients
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        patient_indices = list(range(self.number_of_patients))
        if self.shuffle_patients:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(patient_indices)

        for patient_idx in patient_indices:
            start = patient_idx * self.samples_per_patient
            yield list(range(start, start + self.samples_per_patient))

    def __len__(self):
        return self.number_of_patients


class BraTSDataset(Dataset):
    def __init__(
        self,
        root: Path,
        patient_ids: list[str],
        image_size: int = 240,
        samples_per_patient: int = 4,
        training: bool = True,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.patient_ids = list(patient_ids)
        self.image_size = image_size
        self.samples_per_patient = samples_per_patient
        self.training = training
        self.seed = seed
        self.paths = {}

        for patient_id in self.patient_ids:
            matches = list(self.root.rglob(f"{patient_id}_flair.nii.gz"))
            if matches:
                self.paths[patient_id] = matches[0]

        self.patient_ids = [p for p in self.patient_ids if p in self.paths]
        if not self.patient_ids:
            raise RuntimeError("No FLAIR files found for the supplied patient IDs.")

        # Single-patient cache: batches are grouped by patient by PatientBatchSampler.
        self._cached_patient = None
        self._cached_volume = None
        self._cached_valid_slices = None

    def __len__(self):
        return len(self.patient_ids) * self.samples_per_patient

    def _load_patient(self, patient_id: str):
        if self._cached_patient == patient_id:
            return self._cached_volume, self._cached_valid_slices

        nii = nib.load(str(self.paths[patient_id]))
        volume = nii.get_fdata(dtype=np.float32)

        if volume.ndim != 3:
            raise ValueError(f"{patient_id}: expected a 3D FLAIR volume, got {volume.shape}")

        valid_slices = []
        for z in range(volume.shape[2]):
            slice_2d = volume[:, :, z]
            # Ignore slices with almost no brain signal.
            if np.count_nonzero(slice_2d > 0) >= 500:
                valid_slices.append(z)

        if not valid_slices:
            # Fallback for unusual scans with a different intensity convention.
            valid_slices = list(range(volume.shape[2]))

        self._cached_patient = patient_id
        self._cached_volume = volume
        self._cached_valid_slices = valid_slices
        return volume, valid_slices

    def _make_mask(self, rng: random.Random) -> np.ndarray:
        size = self.image_size
        mask = np.zeros((size, size), dtype=np.float32)

        # 1–3 rectangular missing regions.
        for _ in range(rng.randint(1, 3)):
            rh = rng.randint(max(12, size // 10), max(16, size // 2))
            rw = rng.randint(max(12, size // 10), max(16, size // 2))
            y = rng.randint(0, size - rh)
            x = rng.randint(0, size - rw)
            mask[y:y + rh, x:x + rw] = 1.0

        # Add an irregular brush-like missing region in some samples.
        if rng.random() < 0.65:
            canvas = np.zeros((size, size), dtype=np.uint8)
            x = rng.randint(size // 5, 4 * size // 5)
            y = rng.randint(size // 5, 4 * size // 5)
            points = [(x, y)]
            for _ in range(rng.randint(3, 7)):
                x = int(np.clip(x + rng.randint(-45, 45), 0, size - 1))
                y = int(np.clip(y + rng.randint(-45, 45), 0, size - 1))
                points.append((x, y))

            thickness = rng.randint(8, 24)
            for p1, p2 in zip(points[:-1], points[1:]):
                cv2.line(canvas, p1, p2, 1, thickness=thickness)
                cv2.circle(canvas, p2, thickness // 2, 1, -1)
            mask = np.maximum(mask, canvas.astype(np.float32))

        # Ensure a meaningful but not near-total missing area.
        missing_fraction = float(mask.mean())
        if missing_fraction < 0.03 or missing_fraction > 0.65:
            h = rng.randint(size // 5, size // 2)
            w = rng.randint(size // 5, size // 2)
            y = rng.randint(0, size - h)
            x = rng.randint(0, size - w)
            mask[:] = 0
            mask[y:y + h, x:x + w] = 1.0

        return mask

    def __getitem__(self, index: int):
        patient_index = index // self.samples_per_patient
        patient_id = self.patient_ids[patient_index]
        sample_number = index % self.samples_per_patient

        volume, valid_slices = self._load_patient(patient_id)

        # Fixed random choices for validation/test make comparisons repeatable.
        if self.training:
            rng = random
        else:
            rng = random.Random(self.seed + index * 1009)

        z = rng.choice(valid_slices)
        image = volume[:, :, z].astype(np.float32, copy=True)

        nonzero = image > 0
        if np.any(nonzero):
            low, high = np.percentile(image[nonzero], [1, 99])
            image = np.clip((image - low) / max(float(high - low), 1e-8), 0, 1)
            image[~nonzero] = 0
        else:
            image.fill(0)

        image = cv2.resize(
            image, (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        ).astype(np.float32)

        mask = self._make_mask(rng)
        incomplete = image * (1.0 - mask)

        # [1,H,W] tensors; DataLoader adds the batch dimension.
        return (
            torch.from_numpy(incomplete[None]),
            torch.from_numpy(mask[None]),
            torch.from_numpy(image[None]),
        )


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable local SSIM loss for images normalized to [0,1]."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(prediction, kernel_size=7, stride=1, padding=3)
    mu_y = F.avg_pool2d(target, kernel_size=7, stride=1, padding=3)
    sigma_x = F.avg_pool2d(prediction * prediction, 7, 1, 3) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 7, 1, 3) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(prediction * target, 7, 1, 3) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return 1.0 - ssim_map.mean()


def edge_loss(prediction: torch.Tensor, target: torch.Tensor,
              missing_mask: torch.Tensor) -> torch.Tensor:
    """Sobel edge consistency, evaluated in the missing region."""
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=prediction.dtype, device=prediction.device
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(2, 3)

    pred_x = F.conv2d(prediction, kx, padding=1)
    pred_y = F.conv2d(prediction, ky, padding=1)
    true_x = F.conv2d(target, kx, padding=1)
    true_y = F.conv2d(target, ky, padding=1)

    weight = missing_mask
    denominator = weight.sum().clamp_min(1.0)
    return (
        (torch.abs(pred_x - true_x) + torch.abs(pred_y - true_y)) * weight
    ).sum() / denominator


def reconstruction_loss(prediction, target, mask):
    # Pixel loss is focused on missing pixels; SSIM also sees local context.
    denominator = mask.sum().clamp_min(1.0)
    l1 = (torch.abs(prediction - target) * mask).sum() / denominator
    structural = ssim_loss(prediction, target)
    edges = edge_loss(prediction, target, mask)
    total = 0.70 * l1 + 0.20 * structural + 0.10 * edges
    return total, {"l1": l1.detach(), "ssim_loss": structural.detach(),
                   "edge": edges.detach()}


def calculate_unet_metrics(prediction, target, mask):

    from skimage.metrics import (
        peak_signal_noise_ratio,
        structural_similarity
    )

    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy().astype(bool)

    results = []

    for i in range(len(prediction)):

        pred = prediction[i, 0]
        original = target[i, 0]
        missing = mask[i, 0]

        # Final reconstructed image:
        # Received pixels remain unchanged.
        reconstructed = (
            original * (~missing) +
            pred * missing
        )

        # Full image metrics
        mse = float(np.mean(
            (original - reconstructed) ** 2
        ))

        mae = float(np.mean(
            np.abs(original - reconstructed)
        ))

        psnr = peak_signal_noise_ratio(
            original,
            reconstructed,
            data_range=1.0
        )

        ssim = structural_similarity(
            original,
            reconstructed,
            data_range=1.0
        )

        # Missing-region metrics
        if np.any(missing):

            missing_mse = float(np.mean(
                (original[missing] - reconstructed[missing]) ** 2
            ))

            missing_mae = float(np.mean(
                np.abs(original[missing] - reconstructed[missing])
            ))

            missing_psnr = (
                float("inf") if missing_mse == 0
                else 10 * np.log10(1.0 / missing_mse)
            )

        else:
            missing_mae = np.nan
            missing_psnr = np.nan

        results.append({
            "mse": mse,
            "mae": mae,
            "psnr": float(psnr),
            "ssim": float(ssim),
            "missing_mae": missing_mae,
            "missing_psnr": missing_psnr
        })

    return results


def run_epoch(model, loader, optimizer, scaler, device, training, use_amp, label):
    model.train(training)
    losses = []
    metric_rows = []

    progress = tqdm(loader, desc=label, leave=False)
    for incomplete, mask, target in progress:
        incomplete = incomplete.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(use_amp and device.type == "cuda")
            ):
                prediction = model(incomplete, mask)
                loss, _ = reconstruction_loss(prediction, target, mask)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        losses.append(float(loss.detach().item()))
        if not training:
            metric_rows.extend(
                calculate_unet_metrics(
                prediction.float(),
                target.float(),
                mask.float()
            )
)

        progress.set_postfix(loss=f"{losses[-1]:.4f}")

    mean_loss = float(np.mean(losses)) if losses else float("nan")
    if metric_rows:

        metrics = {}

        for key in metric_rows[0]:

            values = np.array([
                row[key] for row in metric_rows
            ], dtype=float)

            metrics[key] = float(np.nanmean(values))

        return mean_loss, metrics

    return mean_loss, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/RECON/checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-patient", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-patients", type=int, default=0,
                        help="Optional quick-run limit; 0 means use all training patients.")
    parser.add_argument("--max-val-patients", type=int, default=0,
                        help="Optional quick-run limit; 0 means use all validation patients.")
    parser.add_argument("--max-test-patients", type=int, default=0,
                        help="Optional quick-run limit; 0 means use all test patients.")
    args = parser.parse_args()

    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16 (e.g. 224 or 240).")
    if args.samples_per_patient < 1:
        raise ValueError("--samples-per-patient must be at least 1.")

    seed_everything(args.seed)
    root = Path(args.root)
    flair_files = find_flair_files(root)
    patient_ids = sorted({f.name.replace("_flair.nii.gz", "") for f in flair_files})

    split_rng = random.Random(args.seed)
    split_rng.shuffle(patient_ids)
    n = len(patient_ids)
    train_end = int(0.80 * n)
    val_end = int(0.90 * n)
    train_ids = patient_ids[:train_end]
    val_ids = patient_ids[train_end:val_end]
    test_ids = patient_ids[val_end:]

    if args.max_train_patients > 0:
        train_ids = train_ids[:args.max_train_patients]
    if args.max_val_patients > 0:
        val_ids = val_ids[:args.max_val_patients]
    if args.max_test_patients > 0:
        test_ids = test_ids[:args.max_test_patients]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_inpainting_model.pth"
    last_path = output_dir / "last_inpainting_model.pth"
    split_path = output_dir / "patient_split.json"

    with split_path.open("w") as f:
        json.dump({
            "seed": args.seed,
            "train_patients": train_ids,
            "validation_patients": val_ids,
            "test_patients": test_ids,
        }, f, indent=2)

    samples = args.samples_per_patient
    train_set = BraTSDataset(root, train_ids, args.image_size, samples, True, args.seed)
    val_set = BraTSDataset(root, val_ids, args.image_size, samples, False, args.seed)
    test_set = BraTSDataset(root, test_ids, args.image_size, samples, False, args.seed)

    train_sampler = PatientBatchSampler(len(train_set.patient_ids), samples, True, args.seed)
    val_sampler = PatientBatchSampler(len(val_set.patient_ids), samples, False, args.seed)
    test_sampler = PatientBatchSampler(len(test_set.patient_ids), samples, False, args.seed)

    # Keep num_workers=0: the dataset's one-volume cache is intentionally in the main process.
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=0,
                             pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.no_amp
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"AMP enabled: {use_amp}")
    print(f"Patients: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Samples per patient per epoch: {samples}")
    print(f"Image size: {args.image_size}x{args.image_size}")

    model = ResidualUNet(base=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history_path = output_dir / "training_history.csv"

    with history_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "epoch", "train_loss", "validation_loss", "epoch_seconds"
        ])
        writer.writeheader()

        for epoch in range(args.epochs):
            start_time = time.time()
            train_sampler.set_epoch(epoch)

            train_loss, _ = run_epoch(
                model, train_loader, optimizer, scaler, device,
                True, use_amp, f"Epoch {epoch + 1}/{args.epochs} [train]"
            )
            val_loss, _ = run_epoch(
                model, val_loader, optimizer, scaler, device,
                False, use_amp, f"Epoch {epoch + 1}/{args.epochs} [validation]"
            )
            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"time={elapsed / 60:.1f} min"
            )
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "epoch_seconds": round(elapsed, 2),
            })
            csvfile.flush()

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "base": 32,
                "epoch": epoch + 1,
                "validation_loss": val_loss,
                "train_patients": train_ids,
                "validation_patients": val_ids,
                "test_patients": test_ids,
                "image_size": args.image_size,
                "model_version": 3,
                "seed": args.seed,
            }
            torch.save(checkpoint, last_path)

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(checkpoint, best_path)
                print(f"  Saved new best checkpoint: {best_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: validation did not improve for {args.patience} epochs.")
                break

    # Evaluate the selected best-validation checkpoint once on held-out test patients.
    print("\nLoading best checkpoint for held-out test evaluation...")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_metrics = run_epoch(
        model, test_loader, optimizer, scaler, device,
        False, use_amp, "[held-out test]"
    )

    test_csv = output_dir / "test_metrics.csv"

    with test_csv.open("w", newline="") as f:

        fields = [
            "best_epoch",
            "test_loss",
            "mse",
            "mae",
            "psnr",
            "ssim",
            "missing_mae",
            "missing_psnr"
        ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    writer.writerow({
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        **test_metrics
    })

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test metrics: {test_metrics}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Patient split: {split_path}")
    print(f"Training history: {history_path}")
    print(f"Test metrics CSV: {test_csv}")


if __name__ == "__main__":
    main()
