import numpy as np
import csv
from pathlib import Path
from skimage.metrics import (
    mean_squared_error,
    mean_absolute_error,
    peak_signal_noise_ratio,
    structural_similarity
)


def calculate_metrics(original, reconstructed, missing_mask, roi_mask):

    original = np.asarray(original, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)

    missing_mask = np.asarray(missing_mask, dtype=bool)
    roi_mask = np.asarray(roi_mask, dtype=bool)

    data_range = 1.0

    mse = mean_squared_error(original, reconstructed)
    mae = mean_absolute_error(original, reconstructed)

    psnr = peak_signal_noise_ratio(
        original, reconstructed, data_range=data_range
    )

    ssim = structural_similarity(
        original, reconstructed, data_range=data_range
    )

    missing_mse = np.mean(
        (original[missing_mask] - reconstructed[missing_mask]) ** 2
    ) if np.any(missing_mask) else np.nan

    missing_mae = np.mean(
        np.abs(original[missing_mask] - reconstructed[missing_mask])
    ) if np.any(missing_mask) else np.nan

    missing_psnr = (
        float("inf") if missing_mse == 0
        else 10 * np.log10(data_range ** 2 / missing_mse)
    ) if np.any(missing_mask) else np.nan

    roi_mse = np.mean(
        (original[roi_mask] - reconstructed[roi_mask]) ** 2
    ) if np.any(roi_mask) else np.nan

    roi_psnr = (
        float("inf") if roi_mse == 0
        else 10 * np.log10(data_range ** 2 / roi_mse)
    ) if np.any(roi_mask) else np.nan

    return {
        "MSE": mse,
        "MAE": mae,
        "PSNR": psnr,
        "SSIM": ssim,
        "Missing_MSE": missing_mse,
        "Missing_MAE": missing_mae,
        "Missing_PSNR": missing_psnr,
        "ROI_PSNR": roi_psnr
    }