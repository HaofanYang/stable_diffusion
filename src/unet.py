# U-Net: 预测噪声 epsilon_theta(x_t, t) 的网络
import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    # 把标量时间步 t 转成一个向量，跟 Transformer 的位置编码是同一个思路
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (b,)
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (b, dim)


class ResidualBlock(nn.Module):
    # 跟之前 modules.py 里的版本类似，但多了 time embedding 的注入
    def __init__(self, in_channels, out_channels, time_dim, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.act = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_mlp(self.act(t_emb))[:, :, None, None]  # 按通道加到每个像素上
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4),  # 每一层下采样通道数的倍数
        num_groups=8,
        time_dim=256,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder（下采样路径），记录每一层的通道数，供 decoder 做 skip connection
        channels = base_channels
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [channels]
        for mult in channel_mults:
            out_channels = base_channels * mult
            self.down_blocks.append(ResidualBlock(channels, out_channels, time_dim, num_groups))
            channels = out_channels
            skip_channels.append(channels)
            self.downsamples.append(Downsample(channels))

        # Bottleneck
        self.mid_block1 = ResidualBlock(channels, channels, time_dim, num_groups)
        self.mid_block2 = ResidualBlock(channels, channels, time_dim, num_groups)

        # Decoder（上采样路径），跟 encoder 对称，每一层先 upsample 再 concat 对应的 skip
        # 注意 upsample 的通道数要匹配"进来的 h"（上一层的输出），不是这一层 block 的输出
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for mult in reversed(channel_mults):
            self.upsamples.append(Upsample(channels))
            skip_ch = skip_channels.pop()
            out_channels = base_channels * mult
            self.up_blocks.append(ResidualBlock(channels + skip_ch, out_channels, time_dim, num_groups))
            channels = out_channels

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, channels + skip_channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels + skip_channels[0], in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h = self.stem(x)
        skips = [h]
        for block, downsample in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = downsample(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        for block, upsample in zip(self.up_blocks, self.upsamples):
            h = upsample(h)
            skip = skips.pop()
            h = block(torch.cat([h, skip], dim=1), t_emb)

        h = torch.cat([h, skips.pop()], dim=1)
        return self.out(h)
