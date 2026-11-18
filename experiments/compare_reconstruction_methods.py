"""Compare Telea, Navier-Stokes and RECON hybrid recovery on one image.

Run from the repository root:
    python -m experiments.compare_reconstruction_methods
Optionally pass --image path/to/image.png and --loss 0.25.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.image_tiling import divide_into_tiles, reconstruct_from_tiles
from src.network_simulator import simulate_packet_loss
from src.image_reconstruction import create_missing_mask, reconstruct_with_inpainting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/input_images/sample.jpg")
    parser.add_argument("--loss", type=float, default=0.25)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    tiles = divide_into_tiles(image, args.tile_size)
    received, lost = simulate_packet_loss(tiles, args.loss, args.seed)
    if not lost:
        lost = {max(tiles): tiles[max(tiles)]}
        received = {k: v for k, v in tiles.items() if k not in lost}

    incomplete = reconstruct_from_tiles(received, image.shape, dtype=image.dtype)
    mask = create_missing_mask(image.shape, received)
    known = mask == 0
    out_dir = Path("results/reconstruction_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tiles: {len(tiles)} | received: {len(received)} | lost: {len(lost)}")
    for method in ("telea", "ns", "hybrid"):
        result = reconstruct_with_inpainting(incomplete, mask, method=method)
        psnr = peak_signal_noise_ratio(image, result, data_range=255)
        ssim = structural_similarity(image, result, data_range=255)
        missing_values = (mask > 0)
        mse_missing = float(np.mean(
            (image[missing_values].astype(np.float32) -
             result[missing_values].astype(np.float32)) ** 2
        ))
        missing_psnr = float("inf") if mse_missing == 0 else 10*np.log10(255**2/mse_missing)
        exact = bool(np.array_equal(result[known], image[known]))
        cv2.imwrite(str(out_dir / f"{method}.png"), result)
        print(
            f"{method:8s} | PSNR={psnr:.3f} dB | SSIM={ssim:.5f} | "
            f"missing-PSNR={missing_psnr:.3f} dB | received-pixels-exact={exact}"
        )
    cv2.imwrite(str(out_dir / "incomplete.png"), incomplete)
    cv2.imwrite(str(out_dir / "missing_mask.png"), mask)
    cv2.imwrite(str(out_dir / "original.png"), image)
    print(f"Saved comparison images to: {out_dir}")


if __name__ == "__main__":
    main()
