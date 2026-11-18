import cv2
import numpy as np

def tile_features(tile, roi_mask=None, reconstruction=None):
    """Return a compact feature vector; no unavailable ground-truth pixels used."""
    a = np.asarray(tile, dtype=np.float32)
    if a.ndim == 3:
        gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    else:
        gray = a
    if gray.max() > 1.0:
        gray = gray / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx*gx + gy*gy)
    roi_fraction = float(np.mean(roi_mask > 0)) if roi_mask is not None else 0.0
    if reconstruction is None:
        confidence = 0.0
    else:
        confidence = float(np.clip(1.0 - np.std(reconstruction), 0, 1))
    return np.array([
        float(np.mean(gray)), float(np.std(gray)),
        float(np.mean(edge)), roi_fraction, confidence
    ], dtype=np.float32)
