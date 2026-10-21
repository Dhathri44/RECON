import random


def simulate_packet_loss(tiles, loss_probability=0.10, seed=42):
    if not 0 <= loss_probability <= 1:
        raise ValueError("Loss probability must be between 0 and 1.")

    rng = random.Random(seed)

    received = {}
    lost = {}

    for tile_id, tile in tiles.items():

        if rng.random() < loss_probability:
            lost[tile_id] = tile
        else:
            received[tile_id] = tile

    return received, lost