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
evidence. The queue model and expected-delay feature have since changed as well.
Current checkpoints are marked `state_version=3` and `environment_version=3`.
Older checkpoints are rejected rather than silently used with a different policy
state or reward environment.

**Retrain the DQN before relying on its decisions.** Until then, the app will
show the rule-based fallback when an old checkpoint is present.

## Actions

| Action | Meaning |
|---|---|
| 0 | Local reconstruction using multiscale inpainting + edge-aware optimization |
| 1 | Normal ARQ retransmission |
| 2 | QoS-priority ARQ retransmission |

The DQN selects among these actions. ARQ uses packet acknowledgements and bounded
retries. The finite shared egress queue uses priority 0 for ROI, priority 1 for
normal image traffic and priority 2 for best-effort/non-ROI traffic. Three
competing background packets are modelled per retransmission decision. Queue
overflow, retransmission counts and primary-flow completion delay are recorded.
This is a controlled single-link multi-flow approximation, not a full
standards-compliant TCP/IP stack or a validated production QoS scheduler.

The current reward is explicit and component-wise:

`R = 10 Q - 0.25 B - 0.05 min(D, 30) - 2 F + 2 Q_ROI`

where `Q` is normalized tile quality, `B` is RECON wire bytes divided by the
original tile size, `D` is primary-tile completion delay divided by its
relative deadline, `F` indicates retransmission fallback, and `Q_ROI` is the
ROI quality score when ROI ground truth is available for evaluation. The
weights are experimental hyperparameters, not clinically validated values.

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

The bundled version-2 DQN checkpoint is intentionally treated as legacy:
the revised environment now models ROI-aware priority admission, finite queue
capacity and competing background traffic. Retrain the DQN before relying on
its policy. The existing U-Net checkpoint is unchanged.
With the BraTS files extracted into a folder containing matched
`*_flair.nii.gz` and `*_seg.nii.gz` files:

```powershell
python -m experiments.train_dqn_brats --data "D:\BraTS_Dataset\Extracted_Training_Data" --patient-split "results\patient_split.json" --episodes 1500
```

The supplied `results/patient_split.json` reconstructs the previously recorded
875/187/189 patient split from the project metadata. If your U-Net was trained
with a different split, pass the exact `patient_split.json` created by that
training run instead. Training duration depends on hardware. The script now
compares Always Reconstruct, Always Retransmit, DQN and Always QoS on identical
held-out patient/network seeds, and writes per-case results to
`results/brats_test_results.csv`. Review action counts as well as image and
network metrics. The bundled `results/comparison_results.csv` and
`results/comparison_summary.csv` were generated before the queue changes and
must be treated as legacy results; do not combine them with the new experiment.
A completed script run is not, by itself, evidence of clinical validity or
generalization.

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
## Run regression tests

From the project root:

```powershell
python -m pip install pytest
python -m pytest
```

The regression suite checks checkpoint schema handling, queue admission and
priority order, ROI-vs-non-ROI QoS behavior, AIMD rate reduction, source syntax,
and (when `nibabel` is installed) label-independent FLAIR slice selection.

## Optional real TCP transfer check

The simulator and a real socket transfer answer different questions. To verify
actual end-to-end TCP delivery and file integrity on two machines, start the
receiver on one machine:

```powershell
python -m src.real_network receive --host 0.0.0.0 --port 5001 --file received.bin
```

Then send a file from the other machine:

```powershell
python -m src.real_network send --host <RECEIVER_IP> --port 5001 --file image.png
```

This measures real TCP throughput and verifies the received SHA-256 digest.
It does not validate RECON's simulated Selective Repeat ARQ or reproduce its
custom QoS queue; TCP performs its own reliability and congestion control.

The BraTS patient volumes are not distributed in this project archive. The
training and held-out evaluation commands must be run on your locally stored
BraTS dataset; no new medical-image performance numbers are claimed by this
patch.
