"""Receiver-side multiscale and edge-aware image reconstruction."""
import cv2
import numpy as np

from src.optimization import edge_aware_refine


def create_missing_mask(image_shape, received_tiles, tile_size=None):
    h, w = image_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for tile in received_tiles.values():
        y, x = tile["position"]
        th, tw = tile["data"].shape[:2]
        mask[y:y+th, x:x+tw] = 0
    return mask


def _inpaint_channel(image_u8, mask_u8, method, radius):
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(image_u8, mask_u8, float(radius), flag)


def _multiscale_initial(channel, missing, method="telea", levels=3):
    """Coarse-to-fine initialization; only received pixels are trusted."""
    h, w = channel.shape
    pyramid = [(channel, missing)]
    for _ in range(max(0, int(levels) - 1)):
        ph, pw = pyramid[-1][0].shape
        if min(ph, pw) <= 32:
            break
        nh, nw = max(16, ph // 2), max(16, pw // 2)
        small_img = cv2.resize(pyramid[-1][0], (nw, nh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(
            pyramid[-1][1].astype(np.uint8), (nw, nh),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        pyramid.append((small_img, small_mask))

    estimate = None
    for level_img, level_mask in reversed(pyramid):
        if not np.any(level_mask):
            estimate = level_img.copy()
            continue
        if estimate is not None:
            coarse = cv2.resize(
                estimate, (level_img.shape[1], level_img.shape[0]),
                interpolation=cv2.INTER_CUBIC
            )
            local = _inpaint_channel(
                np.clip(level_img * 255.0, 0, 255).astype(np.uint8),
                level_mask.astype(np.uint8) * 255,
                method, radius=max(2, min(5, int(min(level_img.shape) / 64)))
            ).astype(np.float32) / 255.0
            # The coarse estimate carries long-range context; the fine-scale
            # fill restores local boundary detail. Blend only unknown pixels.
            estimate = local
            estimate[level_mask] = (
                0.60 * coarse[level_mask] + 0.40 * local[level_mask]
            )
        else:
            estimate = _inpaint_channel(
                np.clip(level_img * 255.0, 0, 255).astype(np.uint8),
                level_mask.astype(np.uint8) * 255,
                method, radius=max(3, min(7, int(min(level_img.shape) / 40)))
            ).astype(np.float32) / 255.0
        estimate[~level_mask] = level_img[~level_mask]
    return np.clip(estimate, 0.0, 1.0)


def reconstruct_with_inpainting(incomplete_image, missing_mask, method="hybrid",
                                 pyramid_levels=3, refine_iterations=80,
                                 lam=0.10, prior_weight=0.35):
    """Reconstruct missing pixels without modifying any received pixel.

    Supported methods: ``hybrid`` (multiscale Telea + edge-aware optimization),
    ``telea`` and ``ns``. Input dtype/range is preserved on return.
    """
    source = np.asarray(incomplete_image)
    if source.ndim not in (2, 3):
        raise ValueError("Expected a 2-D grayscale or 3-D channel image")
    raw_mask = np.asarray(missing_mask)
    if raw_mask.shape[:2] != source.shape[:2]:
        raise ValueError("missing_mask dimensions must match image")
    missing = raw_mask > 0
    if missing.ndim == 3:
        missing = np.any(missing, axis=2)
    if not np.any(missing):
        return source.copy()

    is_float = np.issubdtype(source.dtype, np.floating)
    if is_float:
        scale = 1.0 if source.size == 0 or float(np.nanmax(source)) <= 1.0 else 255.0
        work = np.clip(source.astype(np.float32) / scale, 0.0, 1.0)
    else:
        scale = 255.0
        work = np.clip(source.astype(np.float32) / 255.0, 0.0, 1.0)

    known = ~missing
    if not np.any(known):
        # No receiver-side evidence exists; do not invent image content.
        return source.copy()

    mode = method.lower()
    if mode not in ("hybrid", "telea", "ns", "navier-stokes"):
        raise ValueError("method must be 'hybrid', 'telea' or 'ns'")
    base_method = "ns" if mode in ("ns", "navier-stokes") else "telea"

    channels = [work] if work.ndim == 2 else [work[..., c] for c in range(work.shape[2])]
    output_channels = []
    for channel in channels:
        if mode == "hybrid":
            initial = _multiscale_initial(
                channel, missing, method="telea", levels=pyramid_levels
            )
            refined = edge_aware_refine(
                initial, channel, known, lam=lam,
                prior_weight=prior_weight, iterations=refine_iterations
            )
        else:
            filled = _inpaint_channel(
                np.clip(channel * 255.0, 0, 255).astype(np.uint8),
                missing.astype(np.uint8) * 255, base_method, radius=5
            ).astype(np.float32) / 255.0
            refined = filled
        refined[known] = channel[known]
        output_channels.append(refined)

    result = output_channels[0] if work.ndim == 2 else np.stack(output_channels, axis=2)
    if is_float:
        result = result * scale
        return result.astype(source.dtype)
    return np.clip(np.rint(result * 255.0), 0, 255).astype(source.dtype)
