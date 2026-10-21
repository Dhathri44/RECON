import cv2
import numpy as np

from src.image_tiling import (
    load_image,
    divide_into_tiles,
    reconstruct_from_tiles
)

from src.network_simulator import simulate_packet_loss

from src.image_reconstruction import (
    create_missing_mask,
    reconstruct_with_inpainting
)


def main():

    # Step 1: Load the original image
    image_path = "data/input_images/sample.jpg"

    image = load_image(image_path)

    print("Original image shape:", image.shape)

    # Step 2: Divide the image into tiles
    tile_size = 32

    tiles = divide_into_tiles(
        image,
        tile_size=tile_size
    )

    print("Total number of tiles:", len(tiles))

    # Step 3: Simulate packet loss
    received, lost = simulate_packet_loss(
        tiles,
        loss_probability=0.10,
        seed=42
    )

    print("Received tiles:", len(received))
    print("Lost tiles:", len(lost))

    actual_loss = (
        len(lost) / len(tiles)
    ) * 100

    print(
        f"Actual packet loss: {actual_loss:.2f}%"
    )

    # Step 4: Reconstruct the incomplete image
    incomplete_image = reconstruct_from_tiles(
        received,
        image.shape
    )

    # Mark missing tiles as black
    for tile in lost.values():

        y, x = tile["position"]

        height, width = tile["data"].shape[:2]

        incomplete_image[
            y:y + height,
            x:x + width
        ] = 0

    # Step 5: Create the missing-region mask
    missing_mask = create_missing_mask(
        image.shape,
        received,
        tile_size=tile_size
    )

    # Step 6: Reconstruct using Telea inpainting
    telea_result = reconstruct_with_inpainting(
        incomplete_image,
        missing_mask,
        method="telea"
    )

    # Step 7: Reconstruct using Navier-Stokes inpainting
    ns_result = reconstruct_with_inpainting(
        incomplete_image,
        missing_mask,
        method="ns"
    )

    # Step 8: Display all results
    cv2.imshow(
        "Original Image",
        image
    )

    cv2.imshow(
        "Image After Packet Loss",
        incomplete_image
    )

    cv2.imshow(
        "Telea Inpainting",
        telea_result
    )

    cv2.imshow(
        "Navier-Stokes Inpainting",
        ns_result
    )

    print("\nPress any key on an image window to exit.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()