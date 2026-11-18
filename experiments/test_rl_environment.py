import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import numpy as np

from src.image_tiling import divide_into_tiles
from src.network_simulator import simulate_packet_loss
from src.rl_environment import ReconstructionTransmissionEnv


# Generate a synthetic grayscale image.
# This is only for testing the environment.
image = np.zeros((128, 128), dtype=np.float32)

for y in range(128):
    for x in range(128):
        image[y, x] = (
            0.5
            + 0.25 * np.sin(x / 10)
            + 0.25 * np.cos(y / 10)
        )

image = np.clip(image, 0, 1).astype(np.float32)


# Divide image into tiles.
tiles = divide_into_tiles(
    image,
    tile_size=32
)


# Simulate initial packet/tile loss.
received, lost = simulate_packet_loss(
    tiles,
    loss_probability=0.25,
    seed=42
)


print("Total tiles:", len(tiles))
print("Initially received:", len(received))
print("Initially lost:", len(lost))


# Create RL environment.
env = ReconstructionTransmissionEnv(
    tiles=tiles,
    lost_tiles=lost,
    seed=42
)


state = env.reset()

print("\nState dimension:", len(state))
print("Initial state:", state)


# Test both actions.
for action in [0, 1]:

    state = env.reset()

    print("\nTesting action:", action)

    next_state, reward, done, info = env.step(action)

    print("Next state dimension:", len(next_state))
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)