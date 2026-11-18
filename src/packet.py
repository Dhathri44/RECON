
from dataclasses import dataclass


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
        # Includes packet ID, tile ID, fragment info,
        # priority field and protocol metadata.
        return 20

    @property
    def total_bytes(self):
        return self.payload_bytes + self.header_bytes
