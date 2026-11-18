import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import numpy as np

from src.packet import (
    packetize_tiles,
    reassemble_tiles
)


# Create a sample image tile.
original = np.random.default_rng(42).random(
    (64, 64),
    dtype=np.float32
)

tiles = {
    0: {
        "position": (0, 0),
        "data": original
    }
}

# Fragment the tile.
packets = packetize_tiles(
    tiles,
    payload_bytes=512
)

print("Original tile shape:", original.shape)
print("Original tile dtype:", original.dtype)
print("Total packets:", len(packets))

# Simulate successful delivery of every packet.
delivered_ids = {
    packet.packet_id for packet in packets
}

# Reassemble.
reconstructed = reassemble_tiles(
    packets,
    delivered_ids
)

result = reconstructed[0]["data"]

print("Reconstructed shape:", result.shape)
print("Reconstructed dtype:", result.dtype)

print(
    "Exact reconstruction:",
    np.array_equal(original, result)
)