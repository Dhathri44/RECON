# RECON
### Optimization-Driven Image Reconstruction and Adaptive Retransmission over Unreliable Networks

RECON is a software-based project that investigates image reconstruction when data is lost during network transmission.

The project simulates packet loss during image transmission and explores image-processing techniques to reconstruct missing regions. It will be extended with optimization algorithms and selective retransmission to improve image quality under network constraints.

## 🎯 Objectives

- Simulate image transmission with packet loss.
- Divide images into smaller tiles and identify missing data.
- Reconstruct missing image regions using image inpainting.
- Compare reconstruction methods.
- Investigate optimization algorithms for image reconstruction.
- Explore selective retransmission under bandwidth and latency constraints.

## 🛠️ Technologies Used

- **Python** — Core implementation
- **OpenCV** — Image processing and inpainting
- **NumPy** — Numerical computations
- **Matplotlib** — Visualization and experimental plots
- **SciPy** — Scientific computing and optimization
- **scikit-image** — Image quality evaluation

## 📂 Project Structure

```text
RECON/
│
├── data/
│   ├── input_images/
│   │   └── sample.jpg
│   └── results/
│
├── src/
│   ├── image_tiling.py
│   ├── network_simulator.py
│   ├── image_reconstruction.py
│   └── optimization.py
│
├── experiments/
│
├── main.py
├── README.md
└── .gitignore
