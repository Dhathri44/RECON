import cv2
import numpy as np


def load_image(path, size=(512, 512)):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    image = cv2.resize(image, size)

    return image


def divide_into_tiles(image, tile_size=64):
    height, width = image.shape[:2]

    tiles = {}
    tile_id = 0

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):

            tile = image[
                y:y + tile_size,
                x:x + tile_size
            ]

            tiles[tile_id] = {
                "position": (y, x),
                "data": tile.copy()
            }

            tile_id += 1

    return tiles


def reconstruct_from_tiles(tiles, image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    for tile in tiles.values():
        y, x = tile["position"]
        data = tile["data"]

        height, width = data.shape[:2]

        image[
            y:y + height,
            x:x + width
        ] = data

    return image