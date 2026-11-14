import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


def evaluate_image(original, reconstructed, missing_mask=None):

    original = np.asarray(original, dtype=np.uint8)
    reconstructed = np.asarray(reconstructed, dtype=np.uint8)

    psnr = peak_signal_noise_ratio(
        original,
        reconstructed,
        data_range=255
    )

    if original.ndim == 2:
        ssim = structural_similarity(
            original,
            reconstructed,
            data_range=255
        )
    else:
        ssim = structural_similarity(
            original,
            reconstructed,
            channel_axis=2,
            data_range=255
        )

    result = {
        "PSNR": psnr,
        "SSIM": ssim
    }

    if missing_mask is not None:

        missing_pixels = missing_mask > 0

        original_pixels = original[missing_pixels]
        reconstructed_pixels = reconstructed[missing_pixels]

        if original_pixels.size > 0:

            mse = np.mean(
                (
                    original_pixels.astype(float)
                    - reconstructed_pixels.astype(float)
                ) ** 2
            )

            if mse == 0:
                missing_psnr = float("inf")
            else:
                missing_psnr = 10 * np.log10(
                    (255 ** 2) / mse
                )

            result["Missing_Region_PSNR"] = missing_psnr

        else:
            result["Missing_Region_PSNR"] = None

    return result