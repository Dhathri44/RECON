import random
from collections import deque
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    bandwidth_mbps: float = 5.0
    latency_ms: float = 40.0
    jitter_ms: float = 10.0
    loss_probability: float = 0.10
    queue_capacity: int = 100
    outage_probability: float = 0.0


class NetworkSimulator:

    def __init__(self, config=None, seed=42):
        self.config = config or NetworkConfig()
        self.rng = random.Random(seed)

        self.queue = deque()
        self.time_s = 0.0

        self.bytes_sent = 0
        self.bytes_delivered = 0
        self.dropped = 0

    def transmit(self, packet):

        c = self.config

        # Count every transmission attempt.
        self.bytes_sent += len(packet.payload)

        # Calculate packet transmission time.
        serialization_ms = (
            len(packet.payload) * 8
        ) / (max(c.bandwidth_mbps, 1e-6) * 1e6) * 1000

        # Calculate network latency and jitter.
        jitter = self.rng.uniform(-c.jitter_ms, c.jitter_ms)

        attempt_delay_ms = (
            max(0.0, c.latency_ms + jitter)
            + serialization_ms
        )

        # Check queue capacity.
        if len(self.queue) >= c.queue_capacity:

            self.dropped += 1

            elapsed_ms = attempt_delay_ms + max(c.latency_ms, 1.0)
            self.time_s += elapsed_ms / 1000

            return False, elapsed_ms

        self.queue.append(packet)
        queued = self.queue.popleft()

        # Simulate packet loss or network outage.
        lost = (
            self.rng.random() < c.outage_probability
            or self.rng.random() < c.loss_probability
        )

        if lost:

            self.dropped += 1

            elapsed_ms = attempt_delay_ms + max(c.latency_ms, 1.0)
            self.time_s += elapsed_ms / 1000

            return False, elapsed_ms

        # Successful packet delivery.
        self.bytes_delivered += len(queued.payload)
        self.time_s += attempt_delay_ms / 1000

        return True, attempt_delay_ms

    def update_conditions(
        self,
        bandwidth_mbps=None,
        latency_ms=None,
        jitter_ms=None,
        loss_probability=None,
        outage_probability=None
    ):

        if bandwidth_mbps is not None:
            self.config.bandwidth_mbps = bandwidth_mbps

        if latency_ms is not None:
            self.config.latency_ms = latency_ms

        if jitter_ms is not None:
            self.config.jitter_ms = jitter_ms

        if loss_probability is not None:
            self.config.loss_probability = loss_probability

        if outage_probability is not None:
            self.config.outage_probability = outage_probability


def simulate_packet_loss(tiles, loss_probability=0.10, seed=42):

    rng = random.Random(seed)

    received = {}
    lost = {}

    for tile_id, tile in tiles.items():

        if rng.random() < loss_probability:
            lost[tile_id] = tile
        else:
            received[tile_id] = tile

    return received, lost