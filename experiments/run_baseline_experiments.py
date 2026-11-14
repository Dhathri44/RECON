import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)



import cv2
import csv
import os

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

from src.evaluation import evaluate_image


IMAGE_PATH = "data/input_images/sample.jpg"

TILE_SIZES = [16, 32, 64]

LOSS_LEVELS = [0.05, 0.10, 0.20, 0.30]

RANDOM_SEEDS = [42, 123, 456]

RESULTS_DIR = "data/results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_experiment():

    image = load_image(IMAGE_PATH)

    results = []

    experiment_number = 1

    for tile_size in TILE_SIZES:

        for loss_probability in LOSS_LEVELS:

            for seed in RANDOM_SEEDS:

                print(
                    f"\nExperiment {experiment_number}"
                )

                print("Tile size:", tile_size)
                print("Packet loss:", loss_probability)
                print("Random seed:", seed)

                tiles = divide_into_tiles(
                    image,
                    tile_size=tile_size
                )


                received, lost = simulate_packet_loss(
                    tiles,
                    loss_probability=loss_probability,
                    seed=seed
                )

                incomplete_image = reconstruct_from_tiles(
                    received,
                    image.shape
                )

                missing_mask = create_missing_mask(
                    image.shape,
                    received,
                    tile_size=tile_size
                )

                telea_result = reconstruct_with_inpainting(
                    incomplete_image,
                    missing_mask,
                    method="telea"
                )

                ns_result = reconstruct_with_inpainting(
                    incomplete_image,
                    missing_mask,
                    method="ns"
                )

                telea_metrics = evaluate_image(
                    image,
                    telea_result,
                    missing_mask
                )

                ns_metrics = evaluate_image(
                    image,
                    ns_result,
                    missing_mask
                )

                actual_loss = (
                    len(lost) / len(tiles)
                ) * 100

                results.append({
                    "Tile_Size": tile_size,
                    "Packet_Loss_Probability": loss_probability,
                    "Actual_Packet_Loss": actual_loss,
                    "Random_Seed": seed,
                    "Total_Tiles": len(tiles),
                    "Lost_Tiles": len(lost),

                    "Telea_PSNR": telea_metrics["PSNR"],
                    "Telea_SSIM": telea_metrics["SSIM"],
                    "Telea_Missing_PSNR":
                        telea_metrics["Missing_Region_PSNR"],

                    "NS_PSNR": ns_metrics["PSNR"],
                    "NS_SSIM": ns_metrics["SSIM"],
                    "NS_Missing_PSNR":
                        ns_metrics["Missing_Region_PSNR"]
                })

                print(
                    "Telea:",
                    telea_metrics
                )

                print(
                    "Navier-Stokes:",
                    ns_metrics
                )

                experiment_number += 1


    output_file = os.path.join(
        RESULTS_DIR,
        "baseline_results.csv"
    )

    with open(
        output_file,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nAll experiments completed.")

    print("Results saved to:", output_file)


if __name__ == "__main__":
    run_experiment()