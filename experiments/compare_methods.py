import sys
import csv
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from src.image_tiling import divide_into_tiles
from src.network_simulator import simulate_packet_loss
from src.rl_environment import ReconstructionTransmissionEnv
from src.dqn_agent import DQNAgent


# --------------------------------------------------
# 1. GENERATE TEST IMAGES
# --------------------------------------------------

def generate_test_image(seed, size=256):

    rng = np.random.default_rng(seed)

    image = np.zeros((size, size), dtype=np.float32)

    yy, xx = np.mgrid[0:size, 0:size]

    image += 0.08

    cx = size // 2
    cy = size // 2

    ellipse = (
        ((xx - cx) / (size * 0.38)) ** 2
        + ((yy - cy) / (size * 0.42)) ** 2
    ) <= 1

    image[ellipse] += 0.30

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

    noise = rng.normal(
        0,
        0.025,
        size=(size, size)
    )

    image += noise.astype(np.float32)

    return np.clip(image, 0, 1).astype(np.float32)


# --------------------------------------------------
# 2. LOAD TRAINED DQN
# --------------------------------------------------

# Create a temporary environment to obtain state dimension.
sample_image = generate_test_image(100)

sample_tiles = divide_into_tiles(
    sample_image,
    tile_size=32
)

_, sample_lost = simulate_packet_loss(
    sample_tiles,
    loss_probability=0.25,
    seed=100
)

sample_env = ReconstructionTransmissionEnv(
    sample_tiles,
    sample_lost,
    seed=100
)

agent = DQNAgent(
    state_dim=13,
    action_dim=3
)

agent.load("models/recon_dqn.pt")

print("Trained DQN model loaded successfully.")


# --------------------------------------------------
# 3. DEFINE EVALUATION POLICIES
# --------------------------------------------------

def always_reconstruct(state):
    return 0


def always_retransmit(state):
    return 1


def always_qos_retransmit(state):
    return 2


def dqn_policy(state):
    return agent.act(state, training=False)


policies = {
    "Always_Reconstruct": always_reconstruct,
    "Always_Retransmit": always_retransmit,
    "Always_QoS_Retransmit": always_qos_retransmit,
    "DQN_3_Action": dqn_policy
}


# --------------------------------------------------
# 4. EVALUATION CONFIGURATION
# --------------------------------------------------

num_scenarios = 30

results = []


# --------------------------------------------------
# 5. EVALUATE EACH SCENARIO
# --------------------------------------------------

for scenario in range(num_scenarios):

    print(
        f"\nEvaluating scenario {scenario + 1}/{num_scenarios}"
    )

    # Generate one test image.
    image = generate_test_image(
        seed=5000 + scenario
    )

    tiles = divide_into_tiles(
        image,
        tile_size=32
    )

    # Generate one missing-tile pattern.
    received, lost = simulate_packet_loss(
        tiles,
        loss_probability=0.25,
        seed=6000 + scenario
    )

    # Each policy gets a separate environment with
    # the same image, lost tiles and initial random seed.
    for policy_name, policy in policies.items():

        env = ReconstructionTransmissionEnv(
            tiles=tiles,
            lost_tiles=lost,
            seed=7000 + scenario
        )

        state = env.reset()

        total_reward = 0.0
        total_bytes = 0
        total_delay = 0.0

        action_counts = {
            0: 0,
            1: 0
        }

        done = False

        while not done:

            action = policy(state)

            action_counts[action] += 1

            next_state, reward, done, info = env.step(action)

            total_reward += reward

            total_bytes += info["bytes_sent"]

            total_delay += info["delay_ms"]

            state = next_state

        # Evaluate the complete reconstructed image.
        reconstructed = env.canvas

        psnr = peak_signal_noise_ratio(
            image,
            reconstructed,
            data_range=1.0
        )

        ssim = structural_similarity(
            image,
            reconstructed,
            data_range=1.0
        )

        results.append({

            "Scenario": scenario + 1,
            "Method": policy_name,
            "Reward": total_reward,
            "PSNR": psnr,
            "SSIM": ssim,
            "Bytes_Sent": total_bytes,
            "Delay_ms": total_delay,
            "Reconstruction_Actions": action_counts[0],
            "Retransmission_Actions": action_counts[1]

        })

        print(
            f"{policy_name:20s} "
            f"Reward={total_reward:.3f} "
            f"PSNR={psnr:.3f} "
            f"SSIM={ssim:.4f} "
            f"Bytes={total_bytes}"
        )


# --------------------------------------------------
# 6. SAVE INDIVIDUAL SCENARIO RESULTS
# --------------------------------------------------

output_path = Path("results/comparison_results.csv")

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(output_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# 7. CALCULATE AVERAGE PERFORMANCE
# --------------------------------------------------

print("\n")
print("=" * 85)
print("FINAL PERFORMANCE COMPARISON")
print("=" * 85)

metric_names = [
    "Reward",
    "PSNR",
    "SSIM",
    "Bytes_Sent",
    "Delay_ms"
]

summary = []

for method in policies:

    method_results = [
        row for row in results
        if row["Method"] == method
    ]

    averages = {}

    for metric in metric_names:

        values = [
            row[metric] for row in method_results
        ]

        averages[metric] = float(np.mean(values))

    summary.append({
        "Method": method,
        **averages
    })

    print(f"\n{method}")

    for metric, value in averages.items():

        print(f"Average {metric}: {value:.4f}")


# --------------------------------------------------
# 8. SAVE SUMMARY
# --------------------------------------------------

summary_path = Path("results/comparison_summary.csv")

with open(summary_path, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=summary[0].keys()
    )

    writer.writeheader()
    writer.writerows(summary)


print("\n")
print("Detailed results:", output_path)
print("Summary results:", summary_path)

print("\nEvaluation completed.")