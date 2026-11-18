"""Regression tests for RECON hybrid reconstruction and trusted-pixel handling."""
import numpy as np
from src.image_reconstruction import reconstruct_with_inpainting
from src.image_tiling import divide_into_tiles
from src.rl_environment import ReconstructionTransmissionEnv


def test_received_pixels_are_unchanged():
    y, x = np.mgrid[:128, :128]
    original = (0.5 + 0.25*np.sin(x/9) + 0.2*np.cos(y/13)).astype(np.float32)
    original = np.clip(original, 0, 1)
    mask = np.zeros((128, 128), np.uint8)
    mask[32:80, 40:88] = 255
    incomplete = original.copy()
    incomplete[mask > 0] = 0.0
    recovered = reconstruct_with_inpainting(incomplete, mask, method="hybrid")
    assert np.array_equal(recovered[mask == 0], original[mask == 0])
    assert np.isfinite(recovered).all()
    assert recovered.min() >= 0 and recovered.max() <= 1


def test_local_estimates_are_not_marked_as_received():
    rng = np.random.default_rng(4)
    original = rng.random((128, 128), dtype=np.float32)
    tiles = divide_into_tiles(original, tile_size=32)
    lost_ids = [0, 1, 4]
    lost = {tile_id: tiles[tile_id] for tile_id in lost_ids}
    env = ReconstructionTransmissionEnv(tiles, lost, seed=9)
    env.reset()
    initially_received = env.received_mask.copy()
    for _ in lost_ids:
        _, _, _, _ = env.step(0)
    assert np.array_equal(env.received_mask, initially_received)
    assert np.array_equal(
        env.canvas[initially_received],
        env.original_image[initially_received]
    )


if __name__ == "__main__":
    test_received_pixels_are_unchanged()
    test_local_estimates_are_not_marked_as_received()
    print("Hybrid reconstruction and trusted-pixel tests passed.")
