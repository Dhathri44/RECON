# RECON — Optimization-Driven Medical Image Recovery

RECON is a research prototype for recovering missing regions of medical images
transmitted over simulated unreliable networks. It combines image processing,
numerical optimization, ARQ retransmission, QoS metadata and a 3-action DQN.

> **Research prototype only. It is not clinically validated and must not be used
> for diagnosis or treatment decisions.** Classical inpainting cannot reliably
> infer anatomy that is completely absent from the received data.

## What changed in this algorithmic update

### 1. Multiscale local reconstruction

`src/image_reconstruction.py` now supports `method="hybrid"`:

1. Builds a coarse-to-fine estimate using the received image and missing-pixel mask.
2. Combines coarse context with finer Telea inpainting.
3. Refines only missing pixels using edge-aware regularization.
4. Restores received pixels exactly before returning the result.

The function still supports `telea` and `ns` for comparison. Input dtype/range is preserved.

### 2. Optimization is now connected to Action 0

`src/optimization.py` retains the original objective and finite-difference gradient
check, and adds `edge_aware_refine()`.

The refinement uses an initial inpainting estimate as a prior and weighted
gradient regularization. Stronger estimated edges receive lower smoothing
weights. Received pixels are hard constraints and are not changed.

This is a classical variational refinement, not a learned medical-image
inpainting model. It cannot recreate unseen tumour structures with certainty.

### 3. Prevent cascading reconstruction errors

`src/rl_environment.py` now distinguishes:
- `received_mask`: pixels actually received from the sender.
- `available_mask`: pixels that can be displayed in the current output, including
  locally reconstructed estimates.

When reconstructing another missing tile, only `received_mask` is treated as
trusted context. A previous hallucinated/reconstructed tile is not silently
promoted to ground truth.

### 4. Evaluation and metadata accounting

The app now reports missing-region PSNR as well as whole-image PSNR/SSIM, recovery
bytes, delay and ROI PSNR when a segmentation mask is supplied.

When BraTS SEG is used as sender-side priority metadata, the simulator accounts
for one priority byte per tile. This is a stated simulation assumption: metadata
is piggybacked on tile headers, so its bytes are counted but no separate
round-trip delay is added.

### 5. DQN checkpoint compatibility

The DQN still has **13 state features and 3 actions**. The receiver-state
semantics changed because reconstructed tiles are no longer treated as received
evidence. New checkpoints are marked `state_version=2`. Old checkpoints are
rejected by the loader rather than silently used with the changed environment.

**Retrain the DQN before relying on its decisions.** Until then, the app will
show the rule-based fallback when an old checkpoint is present.

## Actions

| Action | Meaning |
|---|---|
| 0 | Local reconstruction using multiscale inpainting + edge-aware optimization |
| 1 | Normal ARQ retransmission |
| 2 | QoS-priority ARQ retransmission |

The DQN selects among these actions. ARQ uses packet acknowledgements and bounded
retries. The QoS path attaches a priority class to retransmission packets.
This simulator is an educational model, not a full standards-compliant network
stack or a validated production QoS scheduler.

## Install and run

Use Python 3.10–3.12 (recommended), then from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Upload a matching BraTS FLAIR volume and SEG volume, or a 2-D image. SEG is used
for ROI evaluation and the explicitly modelled sender-side priority metadata;
it is not used as missing MRI pixel content.

## Retrain the 3-action DQN on BraTS

The previous checkpoint is not compatible with the revised state semantics.
With the BraTS files extracted into a folder containing matched
`*_flair.nii.gz` and `*_seg.nii.gz` files:

```powershell
python -m experiments.train_dqn_brats --data "D:\BraTS_Dataset\Extracted_Training_Data" --episodes 1500
```

Training duration depends on hardware. Confirm that a new
`models/recon_dqn.pt` is created and inspect
`results/brats_training_metrics.csv` and
`results/brats_training_metadata.json`. A completed script run is not, by
itself, evidence of generalization; evaluate on held-out patients and multiple
network seeds.

## Compare reconstruction methods

Run:

```powershell
python -m experiments.compare_reconstruction_methods
```

Optional:

```powershell
python -m experiments.compare_reconstruction_methods --image "data/input_images/sample.jpg" --loss 0.25 --tile-size 32 --seed 42
```

The script compares Telea, Navier–Stokes and hybrid recovery using the same
missing-tile pattern. It prints whole-image PSNR/SSIM, missing-region PSNR and
whether received pixels remain exact. Images are saved to
`results/reconstruction_comparison/`.

Run the invariant test:

```powershell
python -m experiments.test_hybrid_reconstruction
```

## Metrics

- PSNR: pixel-wise reconstruction error.
- SSIM: structural similarity.
- Missing-region PSNR: quality only where tiles were lost.
- ROI PSNR: quality inside the supplied ROI mask.
- Recovery bytes: retransmission plus priority-metadata overhead.
- Recovery delay: simulated network time spent on recovery.

Always report the exact dataset split, tile size, loss rate, random seeds and
method settings alongside metrics. Do not compare results generated with
different missing masks.

## Important limitations / next research steps

- Hybrid inpainting is a classical baseline; it is not expected to reconstruct
  large missing anatomical structures as accurately as a suitably trained
  medical-image model.
- The sample-image comparison is only a smoke test. It is not BraTS validation.
- The DQN must be retrained and tested on held-out patients after this update.
- QoS priority is simulated at packet level; this is not a real network
  deployment with competing flows or measured end-to-end QoS guarantees.
- A stronger next step is a trained 2-D/3-D medical-image inpainting model,
  trained only on training patients, with held-out patient evaluation and
  tumour-region metrics. Never use SEG as a hidden reconstruction input.