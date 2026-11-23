# RECON – Medical Image Recovery in Unreliable Networks

RECON is an optimization-driven medical image recovery framework designed to handle missing image regions during transmission over unreliable networks.

It combines image reconstruction techniques with network-level recovery methods to study the trade-off between image quality, communication overhead, and recovery delay. Instead of relying only on retransmission or image reconstruction, RECON uses a Deep Q-Network (DQN) to select a recovery strategy based on image and network conditions.

## 1. Problem

Medical images transmitted over networks can be affected by packet loss, limited bandwidth, latency, and jitter. When parts of an image are lost, the receiver must recover the missing information.

There are two common approaches:

- **Retransmission:** Request the missing data again. This can improve image accuracy but increases communication overhead and delay.
- **Image reconstruction:** Estimate missing regions at the receiver. This reduces retransmission but may introduce reconstruction errors.

RECON combines both approaches and uses adaptive decision-making to select a recovery method.

## 2. Objectives

- Recover missing regions in medical images.
- Preserve image information that was successfully received.
- Reduce unnecessary retransmissions.
- Give priority to important image regions using ROI information.
- Balance image quality, communication overhead, and recovery delay.
- Evaluate different recovery strategies under simulated network conditions.

## 3. System Workflow

The overall processing pipeline is:

1. Load a medical image.
2. Preprocess and normalize the image.
3. Divide the image into smaller tiles.
4. Treat the tiles as individual transmission units.
5. Simulate network conditions such as packet loss, bandwidth, latency, and jitter.
6. Detect missing tiles at the receiver.
7. Extract image-related and network-related features.
8. Select a recovery action using the DQN policy, when a compatible trained checkpoint is available.
9. Reconstruct or retransmit the missing tiles.
10. Assemble the recovered image and evaluate the results.

## 4. Recovery Strategies

RECON supports three recovery actions.

### Action 0 – Local Reconstruction

Missing image regions are estimated at the receiver using image reconstruction methods.

The reconstruction methods include:

- Telea inpainting
- Navier–Stokes inpainting
- Hybrid multiscale reconstruction
- Residual U-Net-based reconstruction
- Edge-aware numerical refinement

The U-Net uses the incomplete image and missing-region mask to estimate the missing image content.

### Action 1 – Normal ARQ

Automatic Repeat reQuest (ARQ) is used to request missing packets again.

The receiver uses acknowledgements and selective retransmission to recover missing data.

### Action 2 – QoS-Prioritized ARQ

Quality of Service (QoS) is used to assign priorities to packets.

Important image regions can be given higher transmission priority so that their missing data is handled before less important regions.

## 5. Models and Techniques

### Image Processing and Reconstruction

- Image preprocessing and normalization
- Image tiling and reassembly
- Missing-region mask generation
- Telea and Navier–Stokes inpainting
- Hybrid multiscale reconstruction

### Numerical Optimization

An edge-aware refinement approach is used to improve reconstructed regions while preserving successfully received image information.

### Residual U-Net

A trained Residual U-Net is used for estimating missing image regions.

Its input contains two channels:

- Incomplete image
- Missing-region mask

The network uses an encoder-decoder structure with skip connections to retain spatial information during reconstruction.

### Deep Reinforcement Learning

A Deep Q-Network (DQN) is used to select among the three recovery actions.

The model uses:

- State dimension: 13
- Number of actions: 3
- Hidden layers: 2
- Neurons per hidden layer: 128
- Discount factor: 0.95

The DQN decides which recovery strategy to use. The U-Net performs image reconstruction; these two models have different roles.

## 6. Dataset

RECON uses the **Brain Tumor Segmentation (BraTS) dataset** for medical image experiments.

The implementation uses:

- **FLAIR MRI:** Input medical image.
- **Segmentation (SEG):** Tumour-region information used for ROI prioritization and evaluation.

### Preprocessing

1. Load the matching FLAIR and segmentation volumes.
2. Select an axial slice.
3. Normalize the FLAIR image intensities.
4. Generate the corresponding ROI mask.
5. Resize the image and mask for the experiment.
6. Divide the image into transmission tiles.

The segmentation mask is used as ROI metadata and for evaluation. It is not supplied to the reconstruction model as missing MRI image content.

The BraTS dataset is not included in this repository and must be obtained separately.

## 7. Network Simulation

RECON evaluates recovery strategies under simulated network conditions.

The application allows the user to configure:

| Parameter | Description |
|---|---|
| Tile size | Size of each image tile |
| Initial tile loss | Fraction of tiles initially lost |
| Bandwidth | Available network bandwidth |
| Latency | Network transmission delay |
| Jitter | Variation in network delay |
| Packet loss | Probability of packet loss |
| Random seed | Controls the simulation randomness |

The simulator provides a controlled environment for comparing recovery strategies. It does not represent a deployment over a real TCP/UDP network.

## 8. Evaluation Metrics

The reconstructed image and communication performance are evaluated using:

| Metric | Purpose |
|---|---|
| PSNR | Measures pixel-level reconstruction quality |
| SSIM | Measures structural similarity |
| Missing-Region PSNR | Evaluates reconstruction quality in missing areas |
| ROI PSNR | Measures quality within the selected ROI |
| Recovery Bytes | Measures communication overhead |
| Recovery Delay | Measures time spent recovering missing data |

These metrics help compare local reconstruction, retransmission, and adaptive recovery.

## 9. Example Experimental Result

The following values are from one demonstrated experimental run, not averages across the entire dataset.

### Image Quality

| Metric | Result |
|---|---:|
| PSNR | 29.750 dB |
| SSIM | 0.960 |
| Missing-Region PSNR | 22.991 dB |
| ROI PSNR | 29.989 dB |

### Communication Performance

| Metric | Result |
|---|---:|
| Recovery Bytes | 15,895 bytes |
| Priority Metadata Overhead | 256 bytes |
| Recovery Delay | 147.306 ms |

### Recovery Action Distribution

| Recovery Action | Tiles Processed |
|---|---:|
| Local Reconstruction | 45 |
| Normal ARQ | 5 |
| QoS-Prioritized ARQ | 4 |
| **Total** | **54** |

Results depend on the input image, network configuration, random seed, and model checkpoint. Further experiments across different images and network conditions are needed to establish general performance.

## 10. Technologies Used

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Image Processing:** OpenCV, NumPy, scikit-image
- **Scientific Computing:** SciPy
- **Medical Image Handling:** nibabel
- **Data Analysis:** Pandas
- **Application Interface:** Streamlit

## 11. Project Structure

The main project components are organized as follows:

```text
RECON/
│
├── app.py
│
├── src/
│   ├── medical_data.py
│   ├── image_tiling.py
│   ├── image_reconstruction.py
│   ├── optimization.py
│   ├── network_simulator.py
│   ├── retransmission.py
│   ├── rl_environment.py
│   ├── dqn_agent.py
│   ├── replay_buffer.py
│   ├── unet_inpainting.py
│   └── evaluation.py
│
├── experiments/
│   ├── train_dqn_brats.py
│   └── other experiment and evaluation scripts
│
├── models/
│   ├── best_inpainting_model.pth
│   └── recon_dqn.pt
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

Model checkpoints and experiment outputs may vary depending on the version of the project being used.

## 12. Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip

### Clone the repository

```bash
git clone https://github.com/Dhathri44/RECON.git
cd RECON
```

### Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Model checkpoints

Place the trained model checkpoints in the `models/` directory:

- `best_inpainting_model.pth` – trained U-Net checkpoint.
- `recon_dqn.pt` – DQN policy checkpoint compatible with the current source code.

The application can use a rule-based fallback if a compatible DQN checkpoint is unavailable. A fallback run should not be interpreted as a trained reinforcement-learning result.

### Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the RECON interface.

## 13. Using the Application

1. Start the Streamlit application.
2. Select the input type:
   - BraTS NIfTI (FLAIR + SEG)
   - 2D image (PNG/JPG)
3. Upload the image and, where applicable, its matching segmentation or ROI mask.
4. Configure tile size and network conditions.
5. Click **RUN RECON**.
6. View the original image, receiver image before recovery, and reconstructed output.
7. Inspect image-quality metrics and the recovery decision trace.
8. Download the reconstructed image or complete run report.

## 14. Limitations

- Network behaviour is simulated rather than tested over a real communication network.
- Reconstruction quality depends on the input image and the trained model.
- ROI prioritization depends on the availability of suitable ROI metadata.
- DQN performance depends on training and checkpoint compatibility.
- Results from individual experiments do not establish general performance across all medical images.
- The framework has not undergone clinical validation.

## 15. Future Work

- Evaluate the framework across more medical images and network configurations.
- Improve DQN policy learning and generalization.
- Study adaptive recovery under more realistic network conditions.
- Improve reconstruction quality in complex missing regions.
- Evaluate communication overhead and recovery time across larger experiments.

## 16. Disclaimer

RECON is an academic research prototype developed to study medical image recovery under simulated network conditions. It is intended for experimentation and educational purposes only. It is not a medical diagnostic system and must not be used for clinical decisions.

---

**Repository:** https://github.com/Dhathri44/RECON
