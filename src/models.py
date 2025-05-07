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
                 dropout_rate: float = 0.2, 
                 bottleneck: bool = True,
                 pool_k: int = 2):
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
        self.bottleneck = bottleneck
        if bottleneck:
            self.pool  = nn.AvgPool2d(pool_k)          # 48×72 → 24×36
            self.unpool = nn.Upsample(scale_factor=pool_k,
                                      mode="bilinear", align_corners=False)

    def forward(self, x):
        # x: [B,T,C,H,W]  -> iterate over time
        if x.ndim == 4: # accidental [B,C,H,W]
            x = x.unsqueeze(1) # → [B,1,C,H,W]
        B, T, C, H, W = x.shape
        h = c = None
        for t in range(T):
            # feat = torch.relu(self.encoder(x[:, t]))
            feat = self.encoder(x[:, t])
            if self.bottleneck:
                feat = self.pool(feat)
            # propagate through stacked ConvLSTM cells
            for i, cell in enumerate(self.lstm_cells):
                if h is None:
                    h = [torch.zeros_like(feat) for _ in range(self.depth)]
                    c = [torch.zeros_like(feat) for _ in range(self.depth)]
                h[i], c[i] = cell(feat if i == 0 else h[i-1], h[i], c[i])
            # final h[-1] is memory after this timestep
        feat = h[-1]                              # last hidden state
        if self.bottleneck:
            feat = self.unpool(feat)              # back to 48×72
        out = self.dropout(self.se(feat))
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

# ---------------------------------------------------------------------
# 4. ConvLSTM + Dilated U‑Net
# ---------------------------------------------------------------------
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, dilation=1):
        super().__init__()
        p = dilation * (k // 2)
        self.conv1 = nn.Conv2d(in_c, out_c, k, padding=p, dilation=dilation)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_c, out_c, k, padding=p, dilation=dilation)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.skip  = nn.Identity() if in_c == out_c else \
                     nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.BatchNorm2d(out_c))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.skip(x))


class TemporalUNetDilated(nn.Module):
    r"""[B,T,C,H,W] → [B,C_out,H,W]"""
    def __init__(self,
                 n_input_channels,
                 n_output_channels,
                 time_hidden=128,
                 unet_init=64,
                 depth=4,
                 dilation=2,
                 dropout_rate=0.2,
                 bottleneck=True,
                 pool_k=2):
        super().__init__()
        # temporal front‑end
        self.enc2d = nn.Conv2d(n_input_channels, time_hidden, 3, padding=1)
        self.lstm  = ConvLSTMCell(time_hidden, time_hidden)
        # optional spatial bottleneck inside LSTM
        self.bottleneck_flag = bottleneck
        if bottleneck:
            self.pool2d = nn.AvgPool2d(pool_k)
            self.up2d   = nn.Upsample(scale_factor=pool_k,
                                      mode="bilinear", align_corners=False)
        # U‑Net with dilated residual blocks
        downs, ups, feats = nn.ModuleList(), nn.ModuleList(), []
        c = time_hidden
        for d in range(depth):
            out_c = unet_init * 2 ** d
            downs.append(DilatedResidualBlock(c, out_c,
                                              dilation=1 if d % 2 == 0 else dilation))
            feats.append(out_c)
            c = out_c
        self.downs = downs
        self.pool  = nn.MaxPool2d(2)
        self.bottleneck = DilatedResidualBlock(c, c * 2, dilation=dilation)
        c = c * 2
        for d in reversed(range(depth)):
            ups.append(UpBlock(c, feats[d]))
            c = feats[d]
        self.ups = ups
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final   = nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        h = c = None                                      # start empty
        for t in range(T):
            z = torch.relu(self.enc2d(x[:, t]))
            if self.bottleneck_flag:                      # pooled to 24×36
                z = self.pool2d(z)
            if h is None:
                h = torch.zeros_like(z)
                c = torch.zeros_like(z)
            elif h.shape[-2:] != z.shape[-2:]:
                # e.g. if z is 24×36 but h is 12×18, resize h,c to z’s spatial dims
                h = nn.functional.interpolate(h, size=z.shape[-2:], mode="nearest")
                c = nn.functional.interpolate(c, size=z.shape[-2:], mode="nearest")
            h, c = self.lstm(z, h, c)
        x = h
        if self.bottleneck_flag:
            x = self.up2d(x)
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

# ---------------------------------------------------------------------
# 5. Spatio-Temporal Transformer (STTransformer)
# ---------------------------------------------------------------------
        

# class TemporalTransformer(nn.Module):
#     """
#     Spatio-temporal attention (token-per-gridcell) in place of ConvLSTM.
#     Inputs:  x [B, T, C_in, H, W]
#     Outputs:   [B, C_out,    H, W]
#     """
#     def __init__(
#         self,
#         n_input_channels:  int,
#         n_output_channels: int,
#         hidden_channels:   int = 128,   # d_model
#         num_layers:        int = 4,
#         num_heads:         int = 8,
#         dropout_rate:      float = 0.1,
#         bottleneck:        bool  = True,
#         pool_k:            int   = 2,
#         max_window:        int   = 512,  # max T for pos-embeddings
#     ):
#         super().__init__()
#         assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"

#         # 1) per-timestep spatial encoder
#         self.spatial_encoder = nn.Sequential(
#             nn.Conv2d(n_input_channels, hidden_channels, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.bottleneck = bottleneck
#         if bottleneck:
#             self.pool   = nn.AvgPool2d(pool_k)   # ↓ spatial (e.g. 48×72 → 24×36)
#             self.unpool = nn.Upsample(
#                 scale_factor=pool_k, mode="bilinear", align_corners=False
#             )

#         # 2) temporal transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model          = hidden_channels,
#             nhead            = num_heads,
#             dim_feedforward  = hidden_channels * 4,
#             dropout          = dropout_rate,
#             norm_first       = True,
#             batch_first      = True,    # expects (B·S, T, E)
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # 3) learnable temporal positional embeddings [max_window, E]
#         self.time_pe = nn.Parameter(torch.randn(max_window, hidden_channels))

#         # 4) output head
#         self.dropout = nn.Dropout2d(dropout_rate)
#         self.se      = SEBlock(hidden_channels)
#         self.head    = nn.Conv2d(hidden_channels, n_output_channels, 1)

#     def forward(self, x):
#         # handle accidental 4-D input
#         if x.ndim == 4:           # [B,C,H,W]
#             x = x.unsqueeze(1)     # → [B,1,C,H,W]

#         B, T, C, H, W = x.shape

#         # 1) encode each timestep spatially
#         frames = []
#         for t in range(T):
#             f = self.spatial_encoder(x[:, t])    # [B, E, H', W']
#             if self.bottleneck:
#                 f = self.pool(f)                  # ↓ spatial
#             frames.append(f)

#         # 2) stack time and flatten space → (B·S, T, E)
#         #    where S = H'·W'
#         feats = torch.stack(frames, dim=1)       # [B, T, E, H', W']
#         _, _, E, Hp, Wp = feats.shape
#         S = Hp * Wp

#         # bring spatial to batch-axis, time to seq-axis
#         feats = feats.permute(0, 3, 4, 1, 2).contiguous()  # [B, H', W', T, E]
#         feats = feats.view(B * S, T, E)                    # [B·S, T, E]

#         # 3) add temporal pos-emb
#         pe = self.time_pe[:T]                              # [T, E]
#         feats = feats + pe.unsqueeze(0)                    # broadcast to [B·S, T, E]

#         # 4) run transformer, take last timestep
#         feats = self.transformer(feats)                    # [B·S, T, E]
#         feats = feats[:, -1]                               # [B·S, E]

#         # 5) reshape back to spatial map
#         feats = feats.view(B, Hp, Wp, E).permute(0, 3, 1, 2)  # [B, E, H', W']
#         if self.bottleneck:
#             feats = self.unpool(feats)                      # ↑ spatial

#         # 6) final head
#         out = self.head(self.dropout(self.se(feats)))      # [B, C_out, H, W]
#         return out