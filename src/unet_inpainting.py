
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()

        self.enc1 = DoubleConv(2, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(
            base * 16, base * 8, 2, stride=2
        )
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(
            base * 8, base * 4, 2, stride=2
        )
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(
            base * 4, base * 2, 2, stride=2
        )
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(
            base * 2, base, 2, stride=2
        )
        self.dec1 = DoubleConv(base * 2, base)

        self.output = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.output(d1))


def reconstruct_with_unet(model, incomplete, missing_mask, device):
    """
    incomplete: numpy image, normalized to [0, 1]
    missing_mask: 1 = missing, 0 = received
    """

    model.eval()

    image = torch.as_tensor(
        incomplete, dtype=torch.float32, device=device
    )[None, None]

    mask = torch.as_tensor(
        missing_mask, dtype=torch.float32, device=device
    )[None, None]

    with torch.no_grad():
        prediction = model(image, mask)

        # Preserve all received pixels exactly.
        reconstructed = (
            image * (1.0 - mask) +
            prediction * mask
        )

    return reconstructed[0, 0].cpu().numpy()


def load_inpainting_model(checkpoint_path, device=None):
    device = device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = ResidualUNet(
        base=checkpoint.get("base", 32)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model
