"""Reproducible packet-level network simulator used by RECON."""
import random
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
    """Single-link store-and-forward approximation with stochastic loss.

    Calls are serialized in simulation time. Queue capacity is retained as a
    configured limit for extensions; this sequential simulator does not claim
    to model simultaneous multi-flow queue contention.
    """
    def __init__(self, config=None, seed=42):
        self.config = config or NetworkConfig()
        self.rng = random.Random(seed)
        self.time_s = 0.0
        self.bytes_sent = 0
        self.bytes_delivered = 0
        self.dropped = 0
        self._validate_config()

    def _validate_config(self):
        c = self.config
        if c.bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps must be positive")
        if c.latency_ms < 0 or c.jitter_ms < 0:
            raise ValueError("latency_ms and jitter_ms cannot be negative")
        if not 0 <= c.loss_probability <= 1:
            raise ValueError("loss_probability must be in [0, 1]")
        if not 0 <= c.outage_probability <= 1:
            raise ValueError("outage_probability must be in [0, 1]")
        if c.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")

    def transmit(self, packet):
        c = self.config
        wire_bytes = int(getattr(packet, "total_bytes", len(packet.payload)))
        self.bytes_sent += wire_bytes

        serialization_ms = wire_bytes * 8.0 / (c.bandwidth_mbps * 1e6) * 1000.0
        jitter = self.rng.uniform(-c.jitter_ms, c.jitter_ms)
        propagation_ms = max(0.0, c.latency_ms + jitter)
        delay_ms = propagation_ms + serialization_ms

        lost = (self.rng.random() < c.outage_probability or
                self.rng.random() < c.loss_probability)
        self.time_s += delay_ms / 1000.0
        if lost:
            self.dropped += 1
            return False, delay_ms

        self.bytes_delivered += wire_bytes
        return True, delay_ms

    def update_conditions(self, bandwidth_mbps=None, latency_ms=None,
                          jitter_ms=None, loss_probability=None,
                          outage_probability=None):
        updates = {
            "bandwidth_mbps": bandwidth_mbps,
            "latency_ms": latency_ms,
            "jitter_ms": jitter_ms,
            "loss_probability": loss_probability,
            "outage_probability": outage_probability,
        }
        for key, value in updates.items():
            if value is not None:
                setattr(self.config, key, value)
        self._validate_config()


def simulate_packet_loss(tiles, loss_probability=0.10, seed=42):
    if not 0 <= loss_probability <= 1:
        raise ValueError("loss_probability must be in [0, 1]")
    rng = random.Random(seed)
    received, lost = {}, {}
    for tile_id, tile in tiles.items():
        (lost if rng.random() < loss_probability else received)[tile_id] = tile
    return received, lost
