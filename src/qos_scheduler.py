
from collections import deque
from dataclasses import dataclass

@dataclass
class QoSMetrics:
    packets_sent: int = 0
    bytes_sent: int = 0
    queue_drops: int = 0
    peak_occupancy: int = 0

class PriorityScheduler:
    """
    Strict priority packet scheduler.

    Priority:
      0 = critical / ROI
      1 = normal / non-ROI
      2 = control / ACK
    """

    def __init__(self, capacity=100):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity

        self.queues = {
            0: deque(),
            1: deque(),
            2: deque()
        }

        self.metrics = {
            0: QoSMetrics(),
            1: QoSMetrics(),
            2: QoSMetrics()
        }

        self.total_occupancy = 0

    def enqueue(self, packet):
        if packet.priority not in self.queues:
            raise ValueError("Invalid packet priority")

        if self.total_occupancy >= self.capacity:
            self.metrics[packet.priority].queue_drops += 1
            return False

        self.queues[packet.priority].append(packet)
        self.total_occupancy += 1

        self.metrics[packet.priority].peak_occupancy = max(
            self.metrics[packet.priority].peak_occupancy,
            self.total_occupancy
        )

        return True

    def dequeue(self):
        for priority in (0, 1, 2):
            if self.queues[priority]:
                packet = self.queues[priority].popleft()
                self.total_occupancy -= 1

                metric = self.metrics[priority]
                metric.packets_sent += 1
                metric.bytes_sent += packet.total_bytes

                return packet

        return None

    def __len__(self):
        return self.total_occupancy

    def get_metrics(self):
        return {
            priority: vars(metric).copy()
            for priority, metric in self.metrics.items()
        }