"""Packet framing and tile reassembly for RECON.

Tile payloads use a small JSON metadata header followed by raw contiguous
NumPy bytes. No pickle deserialization is used.
"""
from dataclasses import dataclass
import json
import struct
import numpy as np

_FRAME = struct.Struct("!I")
_MAX_HEADER = 16 * 1024


@dataclass
class Packet:
    packet_id: int
    tile_id: int
    payload: bytes
    # 0 = critical/ROI, 1 = normal, 2 = control/ACK
    priority: int = 1
    fragment_index: int = 0
    fragment_count: int = 1

    def __post_init__(self):
        if self.priority not in (0, 1, 2):
            raise ValueError("priority must be 0, 1, or 2")
        if self.fragment_count < 1:
            raise ValueError("fragment_count must be positive")
        if not 0 <= self.fragment_index < self.fragment_count:
            raise ValueError("fragment_index must be within fragment_count")
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        self.payload = bytes(self.payload)

    @property
    def payload_bytes(self):
        return len(self.payload)

    @property
    def header_bytes(self):
        # Fixed simulated network header (not the tile's internal JSON header).
        return 20

    @property
    def total_bytes(self):
        return self.payload_bytes + self.header_bytes


def _serialize_tile(tile):
    data = np.asarray(tile["data"])
    if data.dtype.hasobject or data.dtype.kind not in "biuf":
        raise TypeError("Tile data must use a numeric or boolean NumPy dtype")
    if data.ndim not in (2, 3):
        raise ValueError("Tile data must be 2-D or 3-D")
    position = tuple(int(v) for v in tile["position"])
    if len(position) != 2 or min(position) < 0:
        raise ValueError("Tile position must contain two non-negative integers")
    data = np.ascontiguousarray(data)
    metadata = {
        "dtype": data.dtype.str,
        "shape": list(data.shape),
        "position": list(position),
    }
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    if len(header) > _MAX_HEADER:
        raise ValueError("Tile metadata header is too large")
    return _FRAME.pack(len(header)) + header + data.tobytes(order="C")


def _deserialize_tile(payload):
    if len(payload) < _FRAME.size:
        raise ValueError("Truncated tile frame")
    (header_len,) = _FRAME.unpack(payload[:_FRAME.size])
    if header_len <= 0 or header_len > _MAX_HEADER:
        raise ValueError("Invalid tile metadata length")
    end = _FRAME.size + header_len
    if end > len(payload):
        raise ValueError("Truncated tile metadata")
    metadata = json.loads(payload[_FRAME.size:end].decode("utf-8"))
    dtype = np.dtype(metadata["dtype"])
    if dtype.hasobject or dtype.kind not in "biuf":
        raise ValueError("Unsupported tile dtype")
    shape = tuple(int(v) for v in metadata["shape"])
    position = tuple(int(v) for v in metadata["position"])
    if len(shape) not in (2, 3) or any(v <= 0 for v in shape):
        raise ValueError("Invalid tile shape")
    if len(position) != 2 or min(position) < 0:
        raise ValueError("Invalid tile position")
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    raw = payload[end:]
    if len(raw) != expected:
        raise ValueError("Tile payload size does not match metadata")
    return {"data": np.frombuffer(raw, dtype=dtype).reshape(shape).copy(),
            "position": position}


def packetize_tiles(tiles, payload_bytes=1200, priority=1, start_packet_id=0):
    if not isinstance(payload_bytes, int) or payload_bytes <= 0:
        raise ValueError("payload_bytes must be a positive integer")
    if priority not in (0, 1, 2):
        raise ValueError("priority must be 0, 1, or 2")
    packets = []
    packet_id = int(start_packet_id)
    for tile_id, tile in tiles.items():
        if not isinstance(tile, dict) or "data" not in tile or "position" not in tile:
            raise ValueError(f"Tile {tile_id} must contain 'data' and 'position'")
        serialized = _serialize_tile(tile)
        fragments = [serialized[i:i + payload_bytes]
                     for i in range(0, len(serialized), payload_bytes)]
        count = len(fragments)
        for index, fragment in enumerate(fragments):
            packets.append(Packet(
                packet_id=packet_id, tile_id=int(tile_id), payload=fragment,
                priority=priority, fragment_index=index, fragment_count=count
            ))
            packet_id += 1
    return packets


def reassemble_tiles(packets, acknowledged_ids):
    acknowledged = set(acknowledged_ids)
    grouped = {}
    for packet in packets:
        if packet.packet_id not in acknowledged:
            continue
        entry = grouped.setdefault(packet.tile_id, {
            "fragments": {}, "fragment_count": packet.fragment_count
        })
        if entry["fragment_count"] != packet.fragment_count:
            continue
        # Ignore conflicting duplicate fragments rather than silently overwrite.
        old = entry["fragments"].get(packet.fragment_index)
        if old is not None and old != packet.payload:
            entry["fragments"].pop(packet.fragment_index, None)
            continue
        entry["fragments"][packet.fragment_index] = packet.payload

    recovered = {}
    for tile_id, entry in grouped.items():
        count = entry["fragment_count"]
        fragments = entry["fragments"]
        if len(fragments) != count or set(fragments) != set(range(count)):
            continue
        try:
            recovered[tile_id] = _deserialize_tile(
                b"".join(fragments[i] for i in range(count))
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
            continue
    return recovered
