
from dataclasses import dataclass
import numpy as np


@dataclass
class ImageTile:
    tile_id: int
    row: int
    col: int
    height: int
    width: int
    data: np.ndarray


def divide_into_tiles(image, tile_size=32):
    if not isinstance(tile_size, int) or isinstance(tile_size, bool):
        raise TypeError("tile_size must be an integer")

    if not 4 <= tile_size <= 256:
        raise ValueError("tile_size must be between 4 and 256")

    if image is None or image.ndim < 2:
        raise ValueError("Expected a valid 2D or 3D image")

    height, width = image.shape[:2]
    tiles = []
    tile_id = 0

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            tile = image[
                row:min(row + tile_size, height),
                col:min(col + tile_size, width)
            ]

            tiles.append(
                ImageTile(
                    tile_id=tile_id,
                    row=row,
                    col=col,
                    height=tile.shape[0],
                    width=tile.shape[1],
                    data=tile.copy()
                )
            )
            tile_id += 1

    return tiles


def reconstruct_from_tiles(tiles, image_shape):
    output = np.zeros(image_shape, dtype=np.float32)

    for tile in tiles:
        output[
            tile.row:tile.row + tile.height,
            tile.col:tile.col + tile.width
        ] = tile.data

    return output
