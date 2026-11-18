import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_image(original, reconstructed, missing_mask=None, data_range=None):
    a, b = np.asarray(original), np.asarray(reconstructed)
    if a.shape != b.shape:
        raise ValueError("Original and reconstructed images must have same shape")
    if data_range is None:
        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
        if data_range == 0: data_range = 1.0
    psnr = float(peak_signal_noise_ratio(a, b, data_range=data_range))
    kwargs = {"data_range": data_range}
    if a.ndim == 3:
        kwargs["channel_axis"] = -1
    ssim = float(structural_similarity(a, b, **kwargs))
    result = {"PSNR": psnr, "SSIM": ssim}
    if missing_mask is not None:
        m = np.asarray(missing_mask) > 0
        if np.any(m):
            aa, bb = a[m].astype(float), b[m].astype(float)
            mse = float(np.mean((aa-bb)**2))
            result["Missing_Region_PSNR"] = float("inf") if mse == 0 else float(10*np.log10(data_range**2/mse))
        else:
            result["Missing_Region_PSNR"] = None
    return result

def roi_psnr(original, reconstructed, roi_mask, data_range=1.0):
    m = np.asarray(roi_mask) > 0
    if not np.any(m):
        return None
    mse = float(np.mean((np.asarray(original)[m]-np.asarray(reconstructed)[m])**2))
    return float("inf") if mse == 0 else float(10*np.log10(data_range**2/mse))
