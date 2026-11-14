import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.image_tiling import (
    load_image,
    divide_into_tiles,
    reconstruct_from_tiles
)

from src.network_simulator import simulate_packet_loss

from src.image_reconstruction import (
    create_missing_mask,
    reconstruct_with_inpainting
)

from src.optimization import gradient_descent

from src.evaluation import evaluate_image


image = load_image(
    "data/input_images/sample.jpg"
)

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

gray = gray.astype(np.float64) / 255.0



tile_size = 32

tiles = divide_into_tiles(
    gray,
    tile_size
)


received, lost = simulate_packet_loss(
    tiles,
    loss_probability=0.20,
    seed=42
)



incomplete = reconstruct_from_tiles(
    received,
    gray.shape
)

missing_mask = create_missing_mask(
    gray.shape,
    received,
    tile_size
)

observed_mask = 1.0 - (
    missing_mask.astype(np.float64) / 255.0
)



incomplete_uint8 = (
    incomplete * 255
).astype(np.uint8)

telea_result = reconstruct_with_inpainting(
    incomplete_uint8,
    missing_mask,
    method="telea"
)

telea_gray = cv2.cvtColor(
    telea_result,
    cv2.COLOR_BGR2GRAY
) if telea_result.ndim == 3 else telea_result

optimized, history = gradient_descent(
    incomplete,
    observed_mask,
    lam=0.2,
    iterations=500
)

optimized = np.clip(
    optimized,
    0,
    1
)

optimized_uint8 = (
    optimized * 255
).astype(np.uint8)



original_uint8 = (
    gray * 255
).astype(np.uint8)

telea_metrics = evaluate_image(
    original_uint8,
    telea_gray
)

optimization_metrics = evaluate_image(
    original_uint8,
    optimized_uint8
)

print("\n--- Telea Reconstruction ---")
print(telea_metrics)

print("\n--- Optimization Reconstruction ---")
print(optimization_metrics)



plt.figure(figsize=(14, 9))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(incomplete, cmap="gray")
plt.title("After Packet Loss")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(telea_gray, cmap="gray")
plt.title("Telea Reconstruction")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(optimized, cmap="gray")
plt.title("Gradient Descent Reconstruction")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Gradient Descent Convergence")
plt.grid()

plt.tight_layout()
plt.show()