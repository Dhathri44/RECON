"""Residual U-Net architecture and resolution-safe inference."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.output = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.output(d1))


def reconstruct_with_unet(model, incomplete, missing_mask, device,
                          input_size=None):
    """Infer at checkpoint training resolution, then restore original size.

    ``missing_mask`` uses 1 for missing and 0 for received. Received pixels
    are copied from the original-resolution input after resizing back.
    """
    import numpy as np

    image_np = np.asarray(incomplete, dtype=np.float32)
    mask_np = np.asarray(missing_mask, dtype=np.float32)
    if image_np.ndim != 2 or mask_np.shape != image_np.shape:
        raise ValueError("U-Net expects matching 2-D image and mask arrays")
    if not np.isfinite(image_np).all() or not np.isfinite(mask_np).all():
        raise ValueError("U-Net inputs must contain only finite values")
    mask_np = (mask_np > 0).astype(np.float32)
    if not np.any(mask_np):
        return image_np.copy()

    height, width = image_np.shape
    target_size = int(input_size or getattr(model, "input_size", 240))
    if target_size < 16 or target_size % 16:
        raise ValueError("U-Net input_size must be a positive multiple of 16")

    image = torch.from_numpy(image_np)[None, None].to(device)
    mask = torch.from_numpy(mask_np)[None, None].to(device)
    if (height, width) != (target_size, target_size):
        image_model = F.interpolate(image, size=(target_size, target_size),
                                     mode="bilinear", align_corners=False)
        mask_model = F.interpolate(mask, size=(target_size, target_size),
                                   mode="nearest")
    else:
        image_model, mask_model = image, mask

    model.eval()
    with torch.inference_mode():
        prediction = model(image_model, mask_model)
        if prediction.shape[-2:] != (height, width):
            prediction = F.interpolate(prediction, size=(height, width),
                                        mode="bilinear", align_corners=False)
        reconstructed = image * (1.0 - mask) + prediction * mask

    output = reconstructed[0, 0].detach().cpu().numpy()
    output[mask_np == 0] = image_np[mask_np == 0]
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def load_inpainting_model(checkpoint_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid U-Net checkpoint: model_state_dict is missing")
    base = int(checkpoint.get("base", 32))
    model = ResidualUNet(base=base)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.input_size = int(checkpoint.get("image_size", 240))
    model.to(device)
    model.eval()
    return model
