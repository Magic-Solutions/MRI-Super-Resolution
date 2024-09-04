import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

# Define a convolutional block with Conv3D, GroupNorm, and ReLU activation
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super(ConvBlock, self).__init__()
        # 3D Convolution with padding to maintain spatial dimensions
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        # Group Normalization
        self.norm = nn.GroupNorm(groups, out_channels)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

# Define a self-attention block with multi-head attention for 3D data
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionBlock, self).__init__()
        # Multi-head attention layer
        self.attention = MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Group normalization
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x):
        b, c, d, h, w = x.shape
        # Reshape x for multi-head attention (convert 3D to 2D)
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, D*H*W, C)
        # Apply attention
        attn_output, _ = self.attention(x, x, x)
        # Reshape back to the original 3D shape
        attn_output = attn_output.permute(0, 2, 1).view(b, c, d, h, w)
        # Add the attention output to the original input (residual connection) and normalize
        return self.norm(attn_output + x)

# Define a downsampling block for the encoder path
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        # Define the convolutional blocks
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        # Max pooling layer for downsampling
        self.pool = nn.MaxPool3d(2)
        # Attention block
        self.attention = AttentionBlock(out_channels, num_heads=8)
        # Residual connection to skip over the convolutions
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = x + res
        # Downsample the spatial dimensions
        x = self.pool(x)
        return x

# Define an upsampling block for the decoder path
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # Transposed convolution for upsampling
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # Define the convolutional blocks
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        # Attention block
        self.attention = AttentionBlock(out_channels, num_heads=8)
        # Residual connection
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, skip):
        res = self.residual(x)
        x = self.upconv(x)
        # Concatenate the skip connection from the encoder path
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        # Add the residual connection
        x = x + res
        return x

# Define the bottleneck block between the encoder and decoder paths
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        # Define the convolutional blocks
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        # Attention block
        self.attention = AttentionBlock(out_channels, num_heads=8)
        # Residual connection
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        # Add the residual connection
        x = x + res
        return x

# Define the full 3D U-Net model
class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        # Encoder path with downsampling
        self.down1 = DownBlock(1, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = Bottleneck(512, 1024)

        # Decoder path with upsampling
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.up5 = UpBlock(64, 32)  # Additional upsampling block for higher resolution
        self.up6 = UpBlock(32, 16)  # Final upsampling block

        # Final convolution to reduce to a single output channel
        self.final_conv = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        # Forward pass through the encoder path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # Forward pass through the bottleneck
        bn = self.bottleneck(d4)

        # Forward pass through the decoder path
        u1 = self.up1(bn, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.up5(u4, d1)  # Additional upsampling
        u6 = self.up6(u5, d1)  # Final upsampling

        # Final convolution to produce the output
        out = self.final_conv(u6)
        return out

# Instantiate the model
model = UNet3D()