import jax
import jax.numpy as jnp
from flax import linen as nn

# Define the convolutional block with Conv3D, GroupNorm, and ReLU activation
class ConvBlock(nn.Module):
    out_channels: int
    groups: int = 8

    @nn.compact
    def __call__(self, x):
        # 3D Convolution with padding to maintain spatial dimensions
        x = nn.Conv(self.out_channels, kernel_size=(3, 3, 3), padding='same')(x)
        # Group Normalization
        x = nn.GroupNorm(num_groups=self.groups)(x)
        # ReLU activation
        x = nn.relu(x)
        return x

# Define a self-attention block with multi-head attention for 3D data
class AttentionBlock(nn.Module):
    embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(self, x):
        # Get the shape of the input
        b, c, d, h, w = x.shape
        # Reshape the input to 2D (batch, sequence, channels)
        x = jnp.reshape(x, (b, c, d * h * w))  # (B, C, D*H*W)
        x = jnp.swapaxes(x, 1, 2)  # (B, D*H*W, C)
        # Apply multi-head self-attention
        attn_output = nn.SelfAttention(num_heads=self.num_heads, kernel_init=nn.initializers.xavier_uniform(), qkv_features=self.embed_dim)(x)
        # Reshape back to the original 3D shape
        attn_output = jnp.swapaxes(attn_output, 1, 2)  # (B, C, D*H*W)
        attn_output = jnp.reshape(attn_output, (b, c, d, h, w))
        # Add the attention output to the original input (residual connection) and normalize
        x = nn.GroupNorm(num_groups=1)(attn_output + x)
        return x

# Define a downsampling block for the encoder path
class DownBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # Residual connection to skip over the convolutions
        residual = nn.Conv(self.out_channels, kernel_size=(1, 1, 1))(x)
        # Apply two ConvBlocks
        x = ConvBlock(self.out_channels)(x)
        x = ConvBlock(self.out_channels)(x)
        # Apply attention
        x = AttentionBlock(embed_dim=self.out_channels, num_heads=8)(x)
        # Add residual connection
        x += residual
        # Max pooling to downsample the spatial dimensions
        x = nn.max_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        return x

# Define an upsampling block for the decoder path
class UpBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, skip):
        # Residual connection
        residual = nn.Conv(self.out_channels, kernel_size=(1, 1, 1))(x)
        # Transposed convolution for upsampling
        x = nn.ConvTranspose(self.out_channels, kernel_size=(2, 2, 2), strides=(2, 2, 2))(x)
        # Concatenate the skip connection from the encoder path
        x = jnp.concatenate([x, skip], axis=-1)
        # Apply two ConvBlocks
        x = ConvBlock(self.out_channels)(x)
        x = ConvBlock(self.out_channels)(x)
        # Apply attention
        x = AttentionBlock(embed_dim=self.out_channels, num_heads=8)(x)
        # Add residual connection
        x += residual
        return x

# Define the bottleneck block between the encoder and decoder paths
class Bottleneck(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # Residual connection
        residual = nn.Conv(self.out_channels, kernel_size=(1, 1, 1))(x)
        # Apply two ConvBlocks
        x = ConvBlock(self.out_channels)(x)
        x = ConvBlock(self.out_channels)(x)
        # Apply attention
        x = AttentionBlock(embed_dim=self.out_channels, num_heads=8)(x)
        # Add residual connection
        x += residual
        return x

# Define the full 3D U-Net model
class UNet3D(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Encoder path with downsampling
        d1 = DownBlock(64)(x)
        d2 = DownBlock(128)(d1)
        d3 = DownBlock(256)(d2)
        d4 = DownBlock(512)(d3)

        # Bottleneck
        bn = Bottleneck(1024)(d4)

        # Decoder path with upsampling
        u1 = UpBlock(512)(bn, d4)
        u2 = UpBlock(256)(u1, d3)
        u3 = UpBlock(128)(u2, d2)
        u4 = UpBlock(64)(u3, d1)
        u5 = UpBlock(32)(u4, d1)  # Additional upsampling block for higher resolution
        u6 = UpBlock(16)(u5, d1)  # Final upsampling block

        # Final convolution to reduce to a single output channel
        out = nn.Conv(1, kernel_size=(1, 1, 1))(u6)
        return out
