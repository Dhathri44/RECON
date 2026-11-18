from pathlib import Path
import numpy as np

def load_nifti_pair(flair_path, seg_path, slice_index=None, normalize=True):
    """Load a BraTS FLAIR volume and matching segmentation using nibabel.

    Returns a 2-D slice and matching label slice. If slice_index is None,
    the axial slice with the largest annotated tumor area is selected.
    """
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Install nibabel: pip install nibabel") from exc

    flair_img = nib.load(str(flair_path))
    seg_img = nib.load(str(seg_path))
    flair = np.asarray(flair_img.get_fdata(), dtype=np.float32)
    seg = np.asarray(seg_img.get_fdata(), dtype=np.int16)

    if flair.shape != seg.shape:
        raise ValueError(f"Image/mask shape mismatch: {flair.shape} vs {seg.shape}")

    if flair.ndim != 3:
        raise ValueError(f"Expected 3-D NIfTI volumes, got shape {flair.shape}")

    if slice_index is None:
        areas = np.count_nonzero(seg, axis=(0, 1))
        slice_index = int(np.argmax(areas))
    if not 0 <= slice_index < flair.shape[2]:
        raise IndexError("slice_index is outside the volume")

    image = flair[:, :, slice_index]
    mask = seg[:, :, slice_index]

    if normalize:
        nonzero = image > 0
        if np.any(nonzero):
            lo, hi = np.percentile(image[nonzero], [1, 99])
            image = np.clip((image - lo) / max(float(hi-lo), 1e-8), 0, 1)
        else:
            image = np.zeros_like(image, dtype=np.float32)

    return image.astype(np.float32), mask.astype(np.int16), slice_index

def find_brats_pairs(root):
    """Find matching *_flair.nii[.gz] and *_seg.nii[.gz] files recursively."""
    root = Path(root)
    pairs = []
    for flair in sorted(root.rglob("*_flair.nii*")):
        name = flair.name.replace("_flair.nii.gz", "").replace("_flair.nii", "")
        candidates = [flair.with_name(name + "_seg.nii.gz"),
                      flair.with_name(name + "_seg.nii")]
        seg = next((p for p in candidates if p.exists()), None)
        if seg:
            pairs.append((flair, seg))
    return pairs
