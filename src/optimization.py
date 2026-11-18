"""Optimization routines used by RECON's local image-recovery pipeline.

The original quadratic objective is retained for compatibility and gradient
checking. The edge-aware refinement below regularizes only the unknown pixels;
observed pixels are hard constraints and are restored exactly after every step.
"""
import numpy as np


def objective(x, y, mask, lam=0.2):
    # mask=1 for received/observed pixels, 0 for missing pixels.
    data = mask * (x-y)
    dx, dy = x[:,1:] - x[:,:-1], x[1:,:] - x[:-1,:]
    return 0.5*np.sum(data**2) + 0.5*lam*(np.sum(dx**2)+np.sum(dy**2))


def compute_gradient(x, y, mask, lam=0.2):
    grad = mask*(x-y)
    dx, dy = x[:,1:]-x[:,:-1], x[1:,:]-x[:-1,:]
    grad[:,:-1] -= lam*dx
    grad[:,1:] += lam*dx
    grad[:-1,:] -= lam*dy
    grad[1:,:] += lam*dy
    return grad


def gradient_descent(y, mask, lam=0.2, iterations=500, tolerance=1e-7):
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if y.shape != mask.shape:
        raise ValueError("y and mask must have identical shapes")
    x = y.copy()
    observed = y[mask > 0]
    x[mask == 0] = float(np.mean(observed)) if observed.size else 0.0
    learning_rate = 1.0 / (1.0 + 8.0*lam)
    history = []
    for _ in range(iterations):
        grad = compute_gradient(x, y, mask, lam)
        history.append(objective(x, y, mask, lam))
        if np.linalg.norm(grad)/np.sqrt(grad.size) < tolerance:
            break
        x -= learning_rate*grad
    return x, history


def edge_aware_refine(initial, observed, known_mask, lam=0.10,
                      prior_weight=0.35, iterations=80):
    """Refine an initial fill with edge-aware quadratic regularization.

    Args:
        initial: Initial inpainted image, float in [0, 1], HxW.
        observed: Received image, float in [0, 1], HxW.
        known_mask: True exactly where pixels were received.
    """
    x0 = np.asarray(initial, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    known = np.asarray(known_mask, dtype=bool)
    if x0.shape != y.shape or x0.shape != known.shape or x0.ndim != 2:
        raise ValueError("initial, observed and known_mask must be matching 2-D arrays")
    missing = ~known
    if not np.any(missing):
        return y.copy()

    x = x0.copy()
    x[known] = y[known]

    # Guide the regularizer with the multiscale initial estimate. Strong
    # gradients receive lower smoothing weights, protecting visible edges.
    gx = x0[:, 1:] - x0[:, :-1]
    gy = x0[1:, :] - x0[:-1, :]
    wx = np.clip(1.0 / np.sqrt(gx * gx + 0.015**2), 0.25, 5.0)
    wy = np.clip(1.0 / np.sqrt(gy * gy + 0.015**2), 0.25, 5.0)

    # Conservative step for the weighted 2-D Laplacian.
    step = 0.18 / (prior_weight + 8.0 * lam * 5.0 + 1e-8)
    for _ in range(max(1, int(iterations))):
        grad = prior_weight * missing * (x - x0)
        dx = x[:, 1:] - x[:, :-1]
        dy = x[1:, :] - x[:-1, :]
        fx, fy = wx * dx, wy * dy
        grad[:, :-1] -= lam * fx
        grad[:, 1:] += lam * fx
        grad[:-1, :] -= lam * fy
        grad[1:, :] += lam * fy
        x[missing] -= step * grad[missing]
        x[known] = y[known]  # hard constraint: never alter received pixels
    return np.clip(x, 0.0, 1.0)


def check_gradient():
    rng = np.random.default_rng(42)
    x, y = rng.random((5,5)), rng.random((5,5))
    mask = rng.integers(0,2,(5,5)).astype(float)
    eps, lam = 1e-6, 0.2
    analytic = compute_gradient(x,y,mask,lam)
    numeric = np.zeros_like(x)
    for i in range(5):
        for j in range(5):
            xp, xm = x.copy(), x.copy()
            xp[i,j] += eps
            xm[i,j] -= eps
            numeric[i,j] = (objective(xp,y,mask,lam)-objective(xm,y,mask,lam))/(2*eps)
    return float(np.linalg.norm(analytic-numeric))
