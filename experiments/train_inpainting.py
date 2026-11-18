
from pathlib import Path
import argparse
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.unet_inpainting import ResidualUNet


class BraTSDataset(Dataset):
    def __init__(self, root, patients, samples_per_patient=8):
        self.files = []
        self.samples_per_patient = samples_per_patient

        root = Path(root)

        for patient in patients:
            matches = list(
                root.rglob(f"{patient}_flair.nii.gz")
            )

            if matches:
                self.files.append(matches[0])

        if not self.files:
            raise RuntimeError("No FLAIR files found.")

    def __len__(self):
        return len(self.files) * self.samples_per_patient

    def __getitem__(self, index):
        path = self.files[index % len(self.files)]

        volume = nib.load(str(path)).get_fdata(
            dtype=np.float32
        )

        # Random axial slice.
        z = random.randint(0, volume.shape[2] - 1)
        image = volume[:, :, z]

        low, high = np.percentile(image, [1, 99])

        image = np.clip(
            (image - low) / (high - low + 1e-8),
            0, 1
        )

        image = torch.tensor(
            image, dtype=torch.float32
        )[None]

        # Pad to dimensions divisible by 16.
        h, w = image.shape[-2:]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        image = torch.nn.functional.pad(
            image, (0, pad_w, 0, pad_h)
        )

        mask = torch.zeros_like(image)

        # Random rectangular missing regions.
        H, W = image.shape[-2:]

        for _ in range(random.randint(1, 4)):
            rh = random.randint(max(8, H // 10), max(8, H // 2))
            rw = random.randint(max(8, W // 10), max(8, W // 2))

            y = random.randint(0, H - rh)
            x = random.randint(0, W - rw)

            mask[:, y:y + rh, x:x + rw] = 1

        incomplete = image * (1 - mask)

        return incomplete, mask, image


def loss_function(prediction, target, mask):
    # Reconstruction error in missing regions.
    l1 = (
        torch.abs(prediction - target) * mask
    ).sum() / (mask.sum() + 1e-8)

    # Structural similarity approximation using local statistics.
    # This term encourages structural consistency.
    mse = ((prediction - target) ** 2 * mask).sum()
    mse = mse / (mask.sum() + 1e-8)

    return l1 + 0.5 * mse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="models/inpainting_unet.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    root = Path(args.root)

    flair_files = sorted(
        root.rglob("*_flair.nii.gz")
    )

    patient_ids = sorted({
        f.name.replace("_flair.nii.gz", "")
        for f in flair_files
    })

    random.Random(42).shuffle(patient_ids)

    n = len(patient_ids)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_ids = patient_ids[:train_end]
    val_ids = patient_ids[train_end:val_end]
    test_ids = patient_ids[val_end:]

    train_set = BraTSDataset(root, train_ids)
    val_set = BraTSDataset(root, val_ids)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResidualUNet(base=32).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr
    )

    best_val = float("inf")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for incomplete, mask, target in train_loader:
            incomplete = incomplete.to(device)
            mask = mask.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            prediction = model(incomplete, mask)

            loss = loss_function(
                prediction, target, mask
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for incomplete, mask, target in val_loader:
                incomplete = incomplete.to(device)
                mask = mask.to(device)
                target = target.to(device)

                prediction = model(incomplete, mask)

                val_loss += loss_function(
                    prediction, target, mask
                ).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"Train={train_loss:.5f} "
            f"Validation={val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            torch.save({
                "model_state_dict": model.state_dict(),
                "base": 32,
                "epoch": epoch + 1,
                "validation_loss": val_loss,
                "train_patients": train_ids,
                "validation_patients": val_ids,
                "test_patients": test_ids,
                "model_version": 2
            }, output)

    print(f"Best model saved to {output}")


if __name__ == "__main__":
    main()
