import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"]  = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)

    t = cfg.model.type
    if t == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif t == "temporal_cnn":
        model = TemporalCNN(**model_kwargs)
    elif t == "unet_res":
        model = UNetRes(**model_kwargs)
    elif t == "dilated_res":
        model = DilatedResNet(**model_kwargs)
    elif t == "temporal_unet_dilated":
        model = TemporalUNetDilated(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {t}")

    return model


# --- Model Architectures ---

# ----------------------------------------------------------------------
# Utility blocks (re‑use across models)
# ----------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual 2‑D block identical to your original."""
    def __init__(self, in_channels, out_channels, k=3, s=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, k, stride=s, padding=k // 2)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, k, padding=k // 2)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.skip  = nn.Sequential()
        if s != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=s), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return self.relu(out)


class SEBlock(nn.Module):
    """Lightweight squeeze‑and‑excitation (optional)."""
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // r, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False), nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)


# ----------------------------------------------------------------------
# 0. Given SimpleCNN
# ----------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2

        # Final prediction layers
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.dropout(x)
        x = self.final(x)

        return x


# ----------------------------------------------------------------------
# 1. Spatio‑Temporal CNN with ConvLSTM (“temporal_cnn”)
#    ‑ Treat a sliding window of T months as a sequence.
# ----------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, in_c, hid_c, k=3):
        super().__init__()
        padding = k // 2
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, k, padding=padding)

    def forward(self, x, h, c):
        # x: [B, C, H, W]   h, c: [B, Hc, H, W]
        combined = torch.cat([x, h], dim=1)
        gates    = self.conv(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o = map(torch.sigmoid, (i, f, o))
        g       = torch.tanh(g)
        c_next  = f * c + i * g
        h_next  = o * torch.tanh(c_next)
        return h_next, c_next


class TemporalCNN(nn.Module):
    """
    Input shape expected: [B, T, C, H, W]
      B: batch, T: timesteps (window), C: climate variables
    """
    def __init__(self,
                 n_input_channels: int,
                 n_output_channels: int,
                 hidden_channels: int = 64,
                 depth: int = 3,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.depth = depth
        # First layer shared for all timesteps
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # ConvLSTM stack
        self.lstm_cells = nn.ModuleList(
            [ConvLSTMCell(hidden_channels, hidden_channels) for _ in range(depth)]
        )
        self.dropout = nn.Dropout2d(dropout_rate)
        self.se = SEBlock(hidden_channels)
        self.head    = nn.Conv2d(hidden_channels, n_output_channels, 1)

    def forward(self, x):
        # x: [B,T,C,H,W]  -> iterate over time
        if x.ndim == 4: # accidental [B,C,H,W]
            x = x.unsqueeze(1) # → [B,1,C,H,W]
        B, T, C, H, W = x.shape
        h = c = None
        for t in range(T):
            feat = torch.relu(self.encoder(x[:, t]))
            # propagate through stacked ConvLSTM cells
            for i, cell in enumerate(self.lstm_cells):
                if h is None:
                    h = [torch.zeros_like(feat) for _ in range(self.depth)]
                    c = [torch.zeros_like(feat) for _ in range(self.depth)]
                h[i], c[i] = cell(feat if i == 0 else h[i-1], h[i], c[i])
            # final h[-1] is memory after this timestep
        out = self.dropout(self.se(h[-1]))
        return self.head(out)


# ----------------------------------------------------------------------
# 2. Residual U‑Net‑style Encoder/Decoder (“unet_res”)
# ----------------------------------------------------------------------

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ResidualBlock(in_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if odd dims
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = nn.functional.pad(x, [0, skip.size(-1)-x.size(-1),
                                      0, skip.size(-2)-x.size(-2)])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetRes(nn.Module):
    def __init__(self,
                 n_input_channels,
                 n_output_channels,
                 init_dim=64,
                 depth=4,
                 dropout_rate=0.2):
        super().__init__()
        self.downs, self.ups, chs = nn.ModuleList(), nn.ModuleList(), []
        in_c = n_input_channels
        # Encoder
        for d in range(depth):
            out_c = init_dim * 2**d
            self.downs.append(ResidualBlock(in_c, out_c))
            chs.append(out_c)
            in_c = out_c
        self.pool = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = ResidualBlock(in_c, in_c * 2)
        # Decoder
        for d in reversed(range(depth)):
            self.ups.append(UpBlock(in_c * 2, chs[d]))
            in_c = chs[d]
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final   = nn.Conv2d(in_c, n_output_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        x = self.dropout(x)
        return self.final(x)


# ----------------------------------------------------------------------
# 3. Dilated Residual CNN (“dilated_res”)
#    ‑ alternates standard and dilated convs to widen FOV.
# ----------------------------------------------------------------------

class DilatedResNet(nn.Module):
    def __init__(self,
                 n_input_channels,
                 n_output_channels,
                 init_dim=64,
                 depth=6,
                 dropout_rate=0.2):
        super().__init__()
        layers, c = [], init_dim
        layers.append(ResidualBlock(n_input_channels, c))
        for i in range(1, depth):
            # every second block uses dilation 2
            dilation = 2 if i % 2 else 1
            pad      = dilation
            layers.append(
                ResidualBlock(c, c, k=3, s=1)
            )
            # manually replace conv1 with dilated conv
            if dilation > 1:
                layers[-1].conv1 = nn.Conv2d(c, c, 3, padding=pad, dilation=dilation)
        self.features = nn.Sequential(*layers)
        self.dropout  = nn.Dropout2d(dropout_rate)
        self.head     = nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        return self.head(x)

# ----------------------------------------------------------------------
# 4. ConvLSTM + Dilated‑Residual U‑Net (“temporal_unet_dilated”)
# ----------------------------------------------------------------------

class DilatedResidualBlock(nn.Module):
    """Residual block with selectable dilation."""
    def __init__(self, in_c, out_c, k=3, dilation=1):
        super().__init__()
        pad = dilation * (k // 2)
        self.conv1 = nn.Conv2d(in_c, out_c, k, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, k, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.skip = nn.Sequential()
        if in_c != out_c:
            self.skip = nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.skip(x)
        return self.relu(y)


class TemporalUNetDilated(nn.Module):
    """
    Input  : [B, T, C_in, H, W]   (T = temporal window length)
    Output : [B,   C_out, H, W]
    """
    def __init__(self,
                 n_input_channels: int,
                 n_output_channels: int,
                 time_hidden: int = 64,
                 unet_init: int   = 64,
                 depth: int       = 4,
                 dilation: int    = 2,
                 dropout_rate: float = 0.2):
        super().__init__()

        # --- Temporal front‑end -------------------------------------------------
        self.enc2d = nn.Conv2d(n_input_channels, time_hidden, 3, padding=1)
        self.lstm  = ConvLSTMCell(time_hidden, time_hidden)

        # --- U‑Net with dilated residual blocks --------------------------------
        self.downs, self.ups, feats = nn.ModuleList(), nn.ModuleList(), []
        in_c = time_hidden
        for d in range(depth):
            out_c = unet_init * 2 ** d
            self.downs.append(DilatedResidualBlock(in_c, out_c,
                                                   dilation=1 if d % 2 == 0 else dilation))
            feats.append(out_c)
            in_c = out_c
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DilatedResidualBlock(in_c, in_c * 2, dilation=dilation)

        for d in reversed(range(depth)):
            self.ups.append(UpBlock(in_c * 2, feats[d]))
            in_c = feats[d]

        self.dropout = nn.Dropout2d(dropout_rate)
        self.final   = nn.Conv2d(in_c, n_output_channels, 1)

    def forward(self, x):
        if x.ndim == 4: # accidental [B,C,H,W]
            x = x.unsqueeze(1) # → [B,1,C,H,W]
        # --------------- ConvLSTM over the temporal dimension -------------------
        B, T, C, H, W = x.shape
        h = c = torch.zeros(B, self.enc2d.out_channels, H, W, device=x.device)
        for t in range(T):
            z   = torch.relu(self.enc2d(x[:, t]))
            h, c = self.lstm(z, h, c)
        x = h                                # shape [B, time_hidden, H, W]

        # --------------------------- U‑Net path ---------------------------------
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, sk in zip(self.ups, reversed(skips)):
            x = up(x, sk)

        x = self.dropout(x)
        return self.final(x)
