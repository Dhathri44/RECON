import sys
from pathlib import Path

# Add project root to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.packet import packetize_tiles
from src.network_simulator import NetworkSimulator, NetworkConfig
from src.retransmission import SelectiveRepeatARQ

import numpy as np

tiles = {
    0: {
        "position": (0, 0),
        "data": np.ones((64, 64), dtype=np.uint8)
    }
}

packets = packetize_tiles(tiles, payload_bytes=512)

net = NetworkSimulator(
    NetworkConfig(loss_probability=0.15),
    seed=7
)

result = SelectiveRepeatARQ(
    net,
    max_retries=4
).send(packets)

print("Packets:", len(packets))

print("ARQ result:", {
    k: v for k, v in result.items()
    if k != "delivered_ids"
})

print("Bytes sent:", net.bytes_sent)
print("Bytes delivered:", net.bytes_delivered)