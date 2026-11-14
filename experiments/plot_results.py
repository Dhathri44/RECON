import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/results/baseline_results.csv"
)

summary = df.groupby(
    ["Tile_Size", "Packet_Loss_Probability"]
).mean(numeric_only=True).reset_index()

for tile_size in summary["Tile_Size"].unique():

    data = summary[
        summary["Tile_Size"] == tile_size
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        data["Packet_Loss_Probability"] * 100,
        data["Telea_PSNR"],
        marker="o",
        label="Telea"
    )

    plt.plot(
        data["Packet_Loss_Probability"] * 100,
        data["NS_PSNR"],
        marker="s",
        label="Navier-Stokes"
    )

    plt.xlabel("Packet Loss (%)")
    plt.ylabel("PSNR (dB)")

    plt.title(
        f"Reconstruction Quality - Tile Size {tile_size}"
    )

    plt.legend()
    plt.grid(True)

    plt.savefig(
        f"data/results/psnr_tile_{tile_size}.png"
    )

    plt.show()