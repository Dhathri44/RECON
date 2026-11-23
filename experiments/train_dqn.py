import sys
import os
import csv
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import numpy as np

from src.image_tiling import divide_into_tiles
from src.network_simulator import simulate_packet_loss
from src.rl_environment import ReconstructionTransmissionEnv
from src.dqn_agent import DQNAgent


# --------------------------------------------------
# 1. GENERATE SYNTHETIC MEDICAL-IMAGE-LIKE DATA
# --------------------------------------------------

def generate_medical_image(seed, size=256):

    rng = np.random.default_rng(seed)

    image = np.zeros((size, size), dtype=np.float32)

    yy, xx = np.mgrid[0:size, 0:size]

    # Background intensity.
    image += 0.08

    # Elliptical structures.
    cx = size // 2
    cy = size // 2

    ellipse = (
        ((xx - cx) / (size * 0.38)) ** 2
        + ((yy - cy) / (size * 0.42)) ** 2
    ) <= 1

    image[ellipse] += 0.30

    # Gaussian structures resembling regions of varying intensity.
    for _ in range(8):

        x0 = rng.integers(size // 4, 3 * size // 4)
        y0 = rng.integers(size // 4, 3 * size // 4)

        sigma = rng.uniform(5, 20)
        amplitude = rng.uniform(0.10, 0.40)

        gaussian = amplitude * np.exp(
            -(
                (xx - x0) ** 2 + (yy - y0) ** 2
            ) / (2 * sigma ** 2)
        )

        image += gaussian.astype(np.float32)

    # Add small noise.
    noise = rng.normal(
        0,
        0.025,
        size=(size, size)
    )

    image += noise.astype(np.float32)

    return np.clip(image, 0, 1).astype(np.float32)


# --------------------------------------------------
# 2. INITIALIZE DQN AGENT
# --------------------------------------------------

agent = DQNAgent(
    state_dim=13,
    action_dim=3,
    lr=1e-3,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9995,
    target_update=100,
    seed=42
)


# --------------------------------------------------
# 3. TRAINING CONFIGURATION
# --------------------------------------------------

episodes = 500
batch_size = 64

episode_rewards = []
episode_losses = []

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# 4. TRAINING LOOP
# --------------------------------------------------

for ep in range(episodes):

    # Generate a different image in each episode.
    image = generate_medical_image(seed=ep)

    # Divide image into tiles.
    tiles = divide_into_tiles(
        image,
        tile_size=32
    )

    # Generate a different missing-tile pattern.
    received, lost = simulate_packet_loss(
        tiles,
        loss_probability=0.25,
        seed=1000 + ep
    )

    # Create a new environment.
    yy, xx = np.mgrid[0:256, 0:256]
    roi_mask = (((xx - 128) / 24) ** 2 + ((yy - 128) / 18) ** 2 <= 1).astype(np.uint8)
    env = ReconstructionTransmissionEnv(
        tiles=tiles,
        lost_tiles=lost,
        roi_mask=roi_mask,
        roi_priority_map=roi_mask,
        evaluation_roi_mask=roi_mask,
        seed=2000 + ep
    )

    state = env.reset()

    total_reward = 0.0
    losses = []

    done = False

    while not done:

        # Choose action.
        action = agent.act(state, training=True)

        # Execute action.
        next_state, reward, done, info = env.step(action)

        # Store experience.
        agent.buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Train DQN.
        loss = agent.learn(batch_size)

        if loss is not None:
            losses.append(loss)

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    mean_loss = (
        float(np.mean(losses))
        if losses else 0.0
    )

    episode_losses.append(mean_loss)

    # Print progress.
    if (ep + 1) % 25 == 0:

        recent_rewards = episode_rewards[-25:]

        print(
            f"Episode {ep + 1}/{episodes} | "
            f"Average reward: {np.mean(recent_rewards):.3f} | "
            f"Loss: {mean_loss:.5f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )


# --------------------------------------------------
# 5. SAVE TRAINED MODEL
# --------------------------------------------------

agent.save("models/recon_dqn_synthetic.pt")

print("\nSynthetic-data baseline saved: models/recon_dqn_synthetic.pt")


# --------------------------------------------------
# 6. SAVE TRAINING METRICS
# --------------------------------------------------

csv_path = "results/training_metrics.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Episode",
        "Total_Reward",
        "Average_Loss"
    ])

    for i in range(episodes):

        writer.writerow([
            i + 1,
            episode_rewards[i],
            episode_losses[i]
        ])

print("Training metrics saved:", csv_path)

print("\nTraining completed.")