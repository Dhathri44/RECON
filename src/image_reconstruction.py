import cv2
import numpy as np


def create_missing_mask(image_shape, received_tiles, tile_size=64):
    """
    Create a mask where:
    0   = received pixels
    255 = missing pixels
    """
    height, width = image_shape[:2]

    mask = np.ones((height, width), dtype=np.uint8) * 255

    for tile in received_tiles.values():
        y, x = tile["position"]
        tile_height, tile_width = tile["data"].shape[:2]

        mask[
            y:y + tile_height,
            x:x + tile_width
        ] = 0

    return mask


def reconstruct_with_inpainting(
    incomplete_image,
    missing_mask,
    method="telea"
):
    """
    Reconstruct missing regions using OpenCV inpainting.
    """

    # Ensure mask values are either 0 or 255
    mask = (missing_mask > 0).astype(np.uint8) * 255

    if method.lower() == "telea":

        result = cv2.inpaint(
            incomplete_image,
            mask,
            5,
            cv2.INPAINT_TELEA
        )

    elif method.lower() in ("ns", "navier-stokes"):

        result = cv2.inpaint(
            incomplete_image,
            mask,
            5,
            cv2.INPAINT_NS
        )

    else:
        raise ValueError(
            "Invalid method. Choose 'telea' or 'ns'."
        )

    return result