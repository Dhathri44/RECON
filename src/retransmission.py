import random

from src.packet import Packet


class SelectiveRepeatARQ:
    """
    Simplified per-packet ARQ simulation.

    Features:
    - Independent packet retransmission
    - Maximum retry limit
    - ACK transmission simulation
    - Duplicate packet handling
    - Successful packet tracking

    This is an educational simulation, not a complete
    standards-compliant Selective Repeat implementation.
    """

    def __init__(
        self,
        network,
        max_retries=3,
        seed=123,
        simulate_ack=True
    ):

        self.network = network
        self.max_retries = max_retries
        self.rng = random.Random(seed)
        self.simulate_ack = simulate_ack

    def send(self, packets):

        received_ids = set()
        acknowledged_ids = set()

        attempts = 0
        retransmissions = 0
        delays = []

        data_bytes = 0
        ack_bytes = 0

        for packet in packets:

            ack_received = False

            for retry in range(self.max_retries + 1):

                attempts += 1

                if retry > 0:
                    retransmissions += 1

                # Count every attempted wire byte, including failed attempts
                # and the simulated packet header.
                data_bytes += packet.total_bytes
                success, delay = self.network.transmit(packet)

                # Include successful and failed transmission delays.
                delays.append(delay)

                if not success:
                    continue

                received_ids.add(packet.packet_id)

                # Simulate ACK path.
                if self.simulate_ack:

                    ack_packet = Packet(
                        packet_id=-(packet.packet_id + 1),
                        tile_id=packet.tile_id,
                        fragment_index=packet.fragment_index,
                        fragment_count=packet.fragment_count,
                        payload=b"ACK",
                        priority=packet.priority
                    )

                    ack_bytes += ack_packet.total_bytes
                    ack_success, ack_delay = self.network.transmit(ack_packet)

                    # Include ACK transmission delay, even if the ACK is lost.
                    delays.append(ack_delay)

                    if not ack_success:
                        continue

                # Sender receives acknowledgement.
                acknowledged_ids.add(packet.packet_id)
                ack_received = True
                break

            if not ack_received:
                continue

        total_packets = len(packets)

        delivery_ratio = (
            len(received_ids) / total_packets
            if total_packets else 1.0
        )

        acknowledgement_ratio = (
            len(acknowledged_ids) / total_packets
            if total_packets else 1.0
        )

        return {
            "delivered_ids": received_ids,
            "acknowledged_ids": acknowledged_ids,
            "attempts": attempts,
            "retransmissions": retransmissions,
            "mean_delay_ms": (
                sum(delays) / len(delays)
                if delays else None
            ),
            "total_delay_ms": float(sum(delays)),
            "delivery_ratio": delivery_ratio,
            "acknowledgement_ratio": acknowledgement_ratio,
            "data_bytes": data_bytes,
            "ack_bytes": ack_bytes
        }