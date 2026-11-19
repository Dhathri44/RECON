from dataclasses import dataclass
import pickle
import numpy as np


@dataclass
class Packet:
    packet_id: int
    tile_id: int
    payload: bytes

    # 0 = critical, 1 = normal, 2 = control
    priority: int = 1

    fragment_index: int = 0
    fragment_count: int = 1

    @property
    def payload_bytes(self):
        return len(self.payload)

    @property
    def header_bytes(self):
        # Packet ID, tile ID, fragment information,
        # priority and protocol metadata.
        return 20

    @property
    def total_bytes(self):
        return self.payload_bytes + self.header_bytes


def packetize_tiles(tiles, payload_bytes=1200, priority=1):
    """
    Convert image tiles into network packets.

    Each tile is serialized and divided into smaller fragments.

    Parameters
    ----------
    tiles : dict
        Dictionary containing:
        {
            tile_id: {
                "data": numpy_array,
                "position": (y, x)
            }
        }

    payload_bytes : int
        Maximum payload size of each packet.

    priority : int
        Packet priority.

    Returns
    -------
    list
        List of Packet objects.
    """

    if payload_bytes <= 0:
        raise ValueError("payload_bytes must be positive.")

    if priority not in (0, 1, 2):
        raise ValueError("priority must be 0, 1, or 2.")

    packets = []
    packet_id = 0

    for tile_id, tile in tiles.items():

        if "data" not in tile or "position" not in tile:
            raise ValueError(
                f"Tile {tile_id} must contain 'data' and 'position'."
            )

        # Serialize the complete tile, including its image data
        # and original position.
        serialized_tile = pickle.dumps(
            {
                "data": np.asarray(tile["data"]),
                "position": tuple(tile["position"])
            },
            protocol=pickle.HIGHEST_PROTOCOL
        )

        # Divide serialized data into fragments.
        fragments = [
            serialized_tile[i:i + payload_bytes]
            for i in range(0, len(serialized_tile), payload_bytes)
        ]

        fragment_count = len(fragments)

        for fragment_index, fragment in enumerate(fragments):

            packet = Packet(
                packet_id=packet_id,
                tile_id=int(tile_id),
                payload=fragment,
                priority=priority,
                fragment_index=fragment_index,
                fragment_count=fragment_count
            )

            packets.append(packet)
            packet_id += 1

    return packets


def reassemble_tiles(packets, acknowledged_ids):
    """
    Reconstruct image tiles using successfully acknowledged packets.

    A tile is recovered only when all its fragments have been
    successfully acknowledged.

    Parameters
    ----------
    packets : list
        Original list of Packet objects.

    acknowledged_ids : iterable
        Packet IDs successfully acknowledged by the receiver.

    Returns
    -------
    dict
        Recovered tiles in the format:
        {
            tile_id: {
                "data": numpy_array,
                "position": (y, x)
            }
        }
    """

    acknowledged_ids = set(acknowledged_ids)

    # Group acknowledged fragments by tile.
    tile_fragments = {}

    for packet in packets:

        if packet.packet_id not in acknowledged_ids:
            continue

        if packet.tile_id not in tile_fragments:

            tile_fragments[packet.tile_id] = {
                "fragments": {},
                "fragment_count": packet.fragment_count
            }

        entry = tile_fragments[packet.tile_id]

        # Check fragment consistency.
        if entry["fragment_count"] != packet.fragment_count:
            continue

        entry["fragments"][packet.fragment_index] = packet.payload

    recovered_tiles = {}

    for tile_id, entry in tile_fragments.items():

        fragments = entry["fragments"]
        expected_count = entry["fragment_count"]

        # Do not recover incomplete tiles.
        if len(fragments) != expected_count:
            continue

        # Check that every fragment is present.
        if set(fragments.keys()) != set(range(expected_count)):
            continue

        # Reassemble the serialized tile.
        serialized_tile = b"".join(
            fragments[i]
            for i in range(expected_count)
        )

        try:
            tile = pickle.loads(serialized_tile)

            recovered_tiles[tile_id] = {
                "data": np.asarray(tile["data"]),
                "position": tuple(tile["position"])
            }

        except Exception:
            # Ignore corrupted or invalid serialized tiles.
            continue

    return recovered_tiles