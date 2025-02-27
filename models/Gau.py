import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class AdaptiveGaussianConv(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size

        # Parameter predictor for Gaussian kernel
        self.param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, 3, 1),  # sigma, dx, dy
        )

        # Pre-compute grid
        self.register_buffer('grid_x', None)
        self.register_buffer('grid_y', None)
        self._init_grid()

    def _init_grid(self):
        x = torch.arange(self.kernel_size, dtype=torch.float32)
        y = torch.arange(self.kernel_size, dtype=torch.float32)
        y, x = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('grid_x', x)
        self.register_buffer('grid_y', y)

    def generate_kernel(self, sigma, dx, dy):
        center_x = (self.kernel_size - 1) / 2 + dx
        center_y = (self.kernel_size - 1) / 2 + dy

        gaussian = torch.exp(
            -((self.grid_x - center_x) ** 2 + (self.grid_y - center_y) ** 2) /
            (2 * sigma ** 2)
        )
        return gaussian / gaussian.sum()

    def forward(self, x):
        B, C, H, W = x.shape

        # Predict Gaussian parameters
        params = self.param_predictor(x).squeeze(-1).squeeze(-1)
        sigma = F.softplus(params[:, 0])
        dx = torch.tanh(params[:, 1]) * 2
        dy = torch.tanh(params[:, 2]) * 2

        # Generate kernels for batch
        kernels = torch.stack([
            self.generate_kernel(sigma[i], dx[i], dy[i])
            for i in range(B)
        ])

        # Expand kernels for depthwise conv
        kernels = kernels.view(B, 1, 1, self.kernel_size, self.kernel_size)
        kernels = kernels.expand(-1, C, 1, -1, -1)

        # Apply depthwise convolution
        x_reshaped = x.view(1, B * C, H, W)
        kernels_reshaped = kernels.reshape(B * C, 1, self.kernel_size, self.kernel_size)

        out = F.conv2d(
            x_reshaped,
            kernels_reshaped,
            padding=self.kernel_size // 2,
            groups=B * C
        )

        return out.view(B, C, H, W)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, activation=True):
        super().__init__()
        padding = padding or kernel_size // 2

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MultiScaleGaussianConv(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[3, 5, 7, 9]):
        super().__init__()
        self.scales = scales

        # Standard convolution path
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, out_channels, k)
            for k in scales
        ])

        # Gaussian convolution path
        self.gaussian_convs = nn.ModuleList([
            AdaptiveGaussianConv(in_channels, k)
            for k in scales
        ])

        # Channel attention
        total_channels = out_channels * len(scales) * 2  # *2 for both paths
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // 16, 1),
            nn.SiLU(),
            nn.Conv2d(total_channels // 16, total_channels, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.proj = DepthwiseSeparableConv(total_channels, out_channels, 1)

    def forward(self, x):
        # Standard convolution features
        conv_features = [conv(x) for conv in self.convs]

        # Gaussian convolution features
        gaussian_features = [gconv(x) for gconv in self.gaussian_convs]

        # Combine all features
        all_features = conv_features + gaussian_features
        combined = torch.cat(all_features, dim=1)

        # Apply channel attention
        attention = self.channel_attention(combined)
        combined = combined * attention

        # Project to output dimension
        return self.proj(combined)


class GaussianBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 5, 7), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DepthwiseSeparableConv(c1, c_, 1)
        self.cv2 = MultiScaleGaussianConv(c_, c2, scales=k)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_MultiScaleGaussian(nn.Module):
    def __init__(
            self,
            c1: int,
            c2: int,
            n: int = 1,
            shortcut: bool = False,
            g: int = 1,
            e: float = 0.5,
            scales: List[int] = [3, 5, 7, 9]
    ):
        super().__init__()
        self.c = int(c2 * e)

        # Input projection
        self.cv1 = DepthwiseSeparableConv(c1, 2 * self.c, 1)

        # Output projection
        self.cv2 = DepthwiseSeparableConv((2 + n) * self.c, c2, 1)

        # Multi-scale Gaussian bottleneck layers
        self.m = nn.ModuleList([
            GaussianBottleneck(
                self.c, self.c,
                shortcut=shortcut,
                g=g,
                k=scales,
                e=1.0
            ) for _ in range(n)
        ])

        # Feature fusion attention
        self.fusion_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.c * (2 + n), self.c * (2 + n), 1),
            nn.SiLU(),
            nn.Conv2d(self.c * (2 + n), self.c * (2 + n), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Initial feature split
        y = list(self.cv1(x).chunk(2, 1))

        # Multi-scale Gaussian feature extraction
        y.extend(m(y[-1]) for m in self.m)

        # Feature fusion with attention
        combined = torch.cat(y, 1)
        attention = self.fusion_attention(combined)
        combined = combined * attention

        return self.cv2(combined)


def test_model():
    # Test configuration
    in_channels = 64
    out_channels = 128
    batch_size = 4
    height = 32
    width = 32

    # Create model
    model = C2f_MultiScaleGaussian(
        c1=in_channels,
        c2=out_channels,
        n=2,
        shortcut=True,
        scales=[3, 5, 7, 9]
    )

    # Test input
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test inference speed
    import time

    model.eval()
    num_iterations = 100

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    # Timing
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(x)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations * 1000
    print(f"Average inference time: {avg_time:.2f} ms")


if __name__ == '__main__':
    test_model()