import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.packet import packetize_tiles, reassemble_tiles
from src.network_simulator import NetworkSimulator, NetworkConfig
from src.retransmission import SelectiveRepeatARQ


# Create sample image tile
image = np.random.default_rng(42).random(
    (64, 64), dtype=np.float32
)

tiles = {
    0: {
        "position": (0, 0),
        "data": image
    }
}

# Fragment image
packets = packetize_tiles(tiles, payload_bytes=512)

# Configure network
config = NetworkConfig(
    bandwidth_mbps=5,
    latency_ms=40,
    jitter_ms=5,
    loss_probability=0.15
)

network = NetworkSimulator(config, seed=42)

# Perform ARQ
arq = SelectiveRepeatARQ(
    network,
    max_retries=4,
    seed=42,
    simulate_ack=True
)

result = arq.send(packets)

# Reassemble successfully delivered tiles
reconstructed = reassemble_tiles(
    packets,
    result["acknowledged_ids"]
)

print("Total packets:", len(packets))
print("Transmission attempts:", result["attempts"])
print("Retransmissions:", result["retransmissions"])
print("Data delivery ratio:", result["delivery_ratio"])
print("ACK ratio:", result["acknowledgement_ratio"])
print("Mean delay:", result["mean_delay_ms"])
print("Data bytes:", result["data_bytes"])
print("ACK bytes:", result["ack_bytes"])
print("Successfully reconstructed tiles:", len(reconstructed))

if 0 in reconstructed:
    print(
        "Exact reconstruction:",
        np.array_equal(image, reconstructed[0]["data"])
    )
else:
    print("Tile could not be completely reconstructed.")