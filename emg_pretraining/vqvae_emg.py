#!/usr/bin/env python3
import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def nearest_power_of_two(x: float, min_value: int = 1) -> int:
    x = max(float(min_value), float(x))
    return int(2 ** round(math.log2(x)))


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResStack1D(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, n_blocks: int, kernel_size: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResBlock1D(channels, hidden_channels, kernel_size=kernel_size) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Encoder1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        downsample_factor: int,
        n_res_blocks: int = 2,
        res_hidden_channels: Optional[int] = None,
        kernel_size: int = 5,
    ):
        super().__init__()
        if downsample_factor < 1 or (downsample_factor & (downsample_factor - 1)) != 0:
            raise ValueError("downsample_factor must be a power of 2")

        res_hidden_channels = res_hidden_channels or hidden_channels
        n_steps = int(math.log2(downsample_factor))

        layers = []
        c_in = in_channels
        c_hidden = max(hidden_channels // 2, 32)

        for step in range(n_steps):
            c_out = hidden_channels if step == n_steps - 1 else c_hidden
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(inplace=True),
                ]
            )
            c_in = c_out

        layers.extend(
            [
                nn.Conv1d(c_in, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                ResStack1D(hidden_channels, res_hidden_channels, n_res_blocks, kernel_size=3),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMACodebook1D(nn.Module):
    """
    Shared codebook. Input is [N, C_in, T].
    Output quantized tensor is [N, embed_dim, T], and ids are [N, T].
    """

    def __init__(self, in_channels: int, embed_dim: int, n_embed: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=1)
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(embed_dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.proj(x.float()).transpose(1, 2).contiguous()  # [N, T, D]
        flat = z.reshape(-1, self.embed_dim)

        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        onehot = F.one_hot(embed_ind, self.n_embed).type(flat.dtype)
        embed_ind = embed_ind.view(z.shape[0], z.shape[1])  # [N, T]

        quant = self.embed_code(embed_ind)  # [N, T, D]

        if self.training:
            onehot_sum = onehot.sum(0)
            embed_sum = flat.transpose(0, 1) @ onehot
            self.cluster_size.data.mul_(self.decay).add_(onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quant.detach() - z).pow(2).mean()
        quant = z + (quant - z).detach()
        quant = quant.transpose(1, 2).contiguous()  # [N, D, T]
        return quant, diff, embed_ind

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Jitter(nn.Module):
    """
    Time-jitter regularization on latent tokens.
    Expects input [B, C, T_latent].
    """

    def __init__(self, probability: float = 0.12):
        super().__init__()
        self.probability = float(probability)

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        if self.probability <= 0.0:
            return quantized

        out = quantized.clone()
        src = quantized.detach()
        B, C, T = out.shape
        if T <= 1:
            return out

        device = out.device
        rand = torch.rand(T, device=device)
        replace_mask = rand < self.probability

        for t in range(T):
            if not replace_mask[t]:
                continue
            if t == 0:
                neighbor = 1
            elif t == T - 1:
                neighbor = T - 2
            else:
                neighbor = t - 1 if torch.rand(1, device=device).item() < 0.5 else t + 1
            out[:, :, t] = src[:, :, neighbor]
        return out


class Conv1d1x1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualConv1dGLU(nn.Module):
    """
    Non-incremental WaveNet residual block for continuous sequence decoding.
    """

    def __init__(
        self,
        residual_channels: int,
        gate_channels: int,
        kernel_size: int,
        skip_out_channels: int,
        cin_channels: int = -1,
        dropout: float = 0.05,
        dilation: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.causal = causal

        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = ((kernel_size - 1) // 2) * dilation

        self.conv = nn.Conv1d(
            residual_channels,
            gate_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        if cin_channels > 0:
            self.conv1x1c = Conv1d1x1(cin_channels, gate_channels)
        else:
            self.conv1x1c = None

        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_out_channels)

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)

        if self.causal:
            x = x[:, :, :residual.size(-1)]
        elif x.size(-1) != residual.size(-1):
            # defensive crop if padding/length ever drifts
            min_t = min(x.size(-1), residual.size(-1))
            x = x[:, :, :min_t]
            residual = residual[:, :, :min_t]
            if c is not None:
                c = c[:, :, :min_t]

        a, b = x.chunk(2, dim=1)

        if c is not None:
            cond = self.conv1x1c(c)
            ca, cb = cond.chunk(2, dim=1)
            a = a + ca
            b = b + cb

        x = torch.tanh(a) * torch.sigmoid(b)
        s = self.conv1x1_skip(x)
        x = self.conv1x1_out(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, s


class WaveNetDecoder1D(nn.Module):
    """
    WaveNet-style decoder for continuous multichannel EMG reconstruction.

    Input:
        quantized latent: [B, embed_dim, T_latent]
    Output:
        reconstruction:   [B, out_channels, T]

    Design:
    - optional jitter on latent sequence
    - upsample latent to target length T
    - use upsampled latent both:
        1) as local conditioning
        2) to initialize the residual stream
    - gated dilated residual stack
    - final 1x1 projection to output EMG channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample_factor: int,
        residual_channels: int = 128,
        gate_channels: int = 256,
        skip_out_channels: int = 128,
        kernel_size: int = 3,
        n_layers: int = 12,
        n_stacks: int = 3,
        dropout: float = 0.05,
        causal: bool = False,
        use_jitter: bool = False,
        jitter_probability: float = 0.12,
        conditioning_channels: Optional[int] = None,
    ):
        super().__init__()

        if upsample_factor < 1:
            raise ValueError("upsample_factor must be >= 1")
        if n_layers % n_stacks != 0:
            raise ValueError("n_layers must be divisible by n_stacks")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_factor = upsample_factor
        self.use_jitter = use_jitter

        if use_jitter:
            self.jitter = Jitter(jitter_probability)

        cond_channels = conditioning_channels or residual_channels

        # Upsample latent from T_latent -> T
        if upsample_factor == 1:
            self.upsample = nn.Identity()
            upsampled_channels = in_channels
        else:
            n_steps = int(math.log2(upsample_factor))
            if (1 << n_steps) != upsample_factor:
                raise ValueError("upsample_factor must be a power of 2")

            up_layers = []
            c_in = in_channels
            c_mid = max(residual_channels, in_channels)

            for step in range(n_steps):
                c_out = c_mid
                up_layers.extend(
                    [
                        nn.ConvTranspose1d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm1d(c_out),
                        nn.ReLU(inplace=True),
                    ]
                )
                c_in = c_out

            self.upsample = nn.Sequential(*up_layers)
            upsampled_channels = c_in

        # Two projections from upsampled latent:
        #   - one initializes residual stream
        #   - one is used as local conditioning in each block
        self.input_proj = nn.Conv1d(upsampled_channels, residual_channels, kernel_size=1)
        self.cond_proj = nn.Conv1d(upsampled_channels, cond_channels, kernel_size=1)

        layers_per_stack = n_layers // n_stacks
        self.conv_layers = nn.ModuleList()
        for layer in range(n_layers):
            dilation = 2 ** (layer % layers_per_stack)
            self.conv_layers.append(
                ResidualConv1dGLU(
                    residual_channels=residual_channels,
                    gate_channels=gate_channels,
                    kernel_size=kernel_size,
                    skip_out_channels=skip_out_channels,
                    cin_channels=cond_channels,
                    dropout=dropout,
                    dilation=dilation,
                    causal=causal,
                )
            )

        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_out_channels, skip_out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_out_channels, out_channels, kernel_size=1),
        )

    def forward(self, quantized: torch.Tensor, output_length: Optional[int] = None) -> torch.Tensor:
        if quantized.ndim != 3:
            raise ValueError(f"Expected quantized shape [B, D, T_latent], got {tuple(quantized.shape)}")

        q = quantized
        if self.use_jitter and self.training:
            q = self.jitter(q)

        q_up = self.upsample(q)

        if output_length is not None and q_up.size(-1) != output_length:
            q_up = F.interpolate(q_up, size=output_length, mode="linear", align_corners=False)

        x = self.input_proj(q_up)
        c = self.cond_proj(q_up)

        skips = None
        for block in self.conv_layers:
            x, s = block(x, c)
            skips = s if skips is None else (skips + s) * math.sqrt(0.5)

        out = self.final(skips)
        return out


class VQVAE1D(nn.Module):
    """
    Joint-channel VQ-VAE with a WaveNet decoder.

    Input:  [B, C, T]
    Output: [B, C, T]
    ids:    [B, T_latent]
    """

    def __init__(
        self,
        in_channel: int,
        channel: int = 128,
        embed_dim: int = 64,
        n_embed: int = 128,
        decay: float = 0.99,
        tds_blocks: int = 2,
        tds_channels: int = 64,
        kernel_width: int = 5,
        dropout: float = 0.0,
        fs: int = 500,
        codes_per_second: float = 8.0,
        # WaveNet decoder args
        wavenet_residual_channels: int = 128,
        wavenet_gate_channels: int = 256,
        wavenet_skip_channels: int = 128,
        wavenet_kernel_size: int = 3,
        wavenet_layers: int = 12,
        wavenet_stacks: int = 3,
        wavenet_dropout: float = 0.05,
        wavenet_causal: bool = True,
        use_jitter: bool = False,
        jitter_probability: float = 0.12,
    ):
        super().__init__()
        _ = dropout

        if in_channel <= 0:
            raise ValueError("in_channel must be positive")
        if codes_per_second <= 0:
            raise ValueError("codes_per_second must be > 0")

        downsample_factor = nearest_power_of_two(fs / codes_per_second, min_value=1)

        self.in_channel = int(in_channel)
        self.fs = int(fs)
        self.codes_per_second = float(codes_per_second)
        self.downsample_factor = int(downsample_factor)

        # Encoder
        self.encoder = Encoder1D(
            in_channels=self.in_channel,
            hidden_channels=channel,
            downsample_factor=self.downsample_factor,
            n_res_blocks=tds_blocks,
            res_hidden_channels=tds_channels,
            kernel_size=kernel_width,
        )

        # Quantizer
        self.codebook = EMACodebook1D(
            in_channels=channel,
            embed_dim=embed_dim,
            n_embed=n_embed,
            decay=decay,
        )

        # WaveNet decoder
        self.decoder = WaveNetDecoder1D(
            in_channels=embed_dim,
            out_channels=self.in_channel,
            upsample_factor=self.downsample_factor,
            residual_channels=wavenet_residual_channels,
            gate_channels=wavenet_gate_channels,
            skip_out_channels=wavenet_skip_channels,
            kernel_size=wavenet_kernel_size,
            n_layers=wavenet_layers,
            n_stacks=wavenet_stacks,
            dropout=wavenet_dropout,
            causal=wavenet_causal,
            use_jitter=use_jitter,
            jitter_probability=jitter_probability,
            conditioning_channels=wavenet_residual_channels,
        )

    def _check_input(self, x: torch.Tensor) -> tuple[int, int, int]:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B, C, T], got {tuple(x.shape)}")
        b, c, t = x.shape
        if c != self.in_channel:
            raise ValueError(f"Expected {self.in_channel} channels, got {c}")
        return b, c, t

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._check_input(x)

        h = self.encoder(x)             # [B, H, T_latent]
        q, diff, ids = self.codebook(h) # [B, D, T_latent], [B, T_latent]

        return {
            "quant": q,
            "ids": ids,
            "diff": diff,
        }

    def decode(self, encoded: Dict[str, torch.Tensor] | torch.Tensor, output_length: Optional[int] = None) -> torch.Tensor:
        if isinstance(encoded, dict):
            q = encoded["quant"]
        else:
            q = encoded

        if q.ndim != 3:
            raise ValueError(f"Expected quant to have shape [B, D, T_latent], got {tuple(q.shape)}")

        out = self.decoder(q, output_length=output_length)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, t = self._check_input(x)
        encoded = self.encode(x)
        out = self.decode(encoded, output_length=t)
        latent_loss = encoded["diff"]
        return out, latent_loss

    def encode_ids(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x)
        return encoded["ids"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = VQVAE1D(
        in_channel=8,
        channel=128,
        embed_dim=64,
        n_embed=128,
        fs=500,
        codes_per_second=8.0,
        wavenet_residual_channels=128,
        wavenet_gate_channels=256,
        wavenet_skip_channels=128,
        wavenet_layers=12,
        wavenet_stacks=3,
        wavenet_kernel_size=3,
        wavenet_dropout=0.05,
        wavenet_causal=True,
        use_jitter=True,
        jitter_probability=0.12,
    ).to(device)

    x = torch.randn(2, 8, 512, device=device)
    y, latent = net(x)
    ids = net.encode_ids(x)

    print("input:", x.shape)
    print("recon:", y.shape)
    print("ids:", ids.shape)
    print("latent loss:", float(latent))