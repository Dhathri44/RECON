
import numpy as np

from src.image_tiling import (
    divide_into_tiles,
    reconstruct_from_tiles
)

from src.packet import Packet
from src.qos_scheduler import PriorityScheduler

def test_tile_sizes():
    image = np.random.rand(253, 251).astype(np.float32)

    for size in [8, 10, 16, 24, 32, 48, 64]:
        tiles = divide_into_tiles(image, size)
        reconstructed = reconstruct_from_tiles(
            tiles, image.shape
        )

        assert np.array_equal(image, reconstructed)

        print(f"Tile size {size}: PASS")

def test_priority():
    scheduler = PriorityScheduler(capacity=10)

    scheduler.enqueue(Packet(1, 1, b"normal", priority=1))
    scheduler.enqueue(Packet(2, 2, b"critical", priority=0))
    scheduler.enqueue(Packet(3, 3, b"control", priority=2))

    first = scheduler.dequeue()
    second = scheduler.dequeue()
    third = scheduler.dequeue()

    assert first.priority == 0
    assert second.priority == 1
    assert third.priority == 2

    print("Priority scheduling: PASS")

if __name__ == "__main__":
    test_tile_sizes()
    test_priority()