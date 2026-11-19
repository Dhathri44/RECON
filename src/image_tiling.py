import numpy as np


def divide_into_tiles(image, tile_size=32):
    """
    Divide an image into tiles.

    Returns:
        Dictionary:
        {
            tile_id: {
                "data": numpy array,
                "position": (row, col)
            }
        }
    """

    if not isinstance(tile_size, int) or isinstance(tile_size, bool):
        raise TypeError("tile_size must be an integer")

    if not 4 <= tile_size <= 256:
        raise ValueError("tile_size must be between 4 and 256")

    if image is None or image.ndim < 2:
        raise ValueError("Expected a valid 2D or 3D image")

    height, width = image.shape[:2]

    tiles = {}
    tile_id = 0

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):

            tile = image[
                row:min(row + tile_size, height),
                col:min(col + tile_size, width)
            ]

            tiles[tile_id] = {
                "data": tile.copy(),
                "position": (row, col)
            }

            tile_id += 1

    return tiles


def reconstruct_from_tiles(tiles, image_shape):
    """
    Reconstruct an image from the tile dictionary.
    """

    output = np.zeros(image_shape, dtype=np.float32)

    for tile in tiles.values():

        row, col = tile["position"]
        data = tile["data"]

        h, w = data.shape[:2]

        output[row:row + h, col:col + w] = data

    return output