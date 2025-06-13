import torch
import torch.nn as nn
from omegaconf import DictConfig
import torch.nn.functional as F


def get_model(cfg: DictConfig):
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}

    # 1) Base raw-variable channels
    C0 = len(cfg.data.input_vars)

    # 2) Derived features: raw + Δ + cumsum
    if getattr(cfg.data, "derived_features", True):
        C1 = C0 * 3
    else:
        C1 = C0

    # 3) Month encoding (sin & cos)
    if getattr(cfg.data, "month_encoding", True):
        C2 = C1 + 2
    else:
        C2 = C1

    # 4) Spatial smoothing duplicates all existing channels
    if getattr(cfg.data, "spatial_smoothing", False):
        C3 = C2 * 2
    else:
        C3 = C2

    model_kwargs["n_input_channels"]  = C3
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)

    t = cfg.model.type
    if t == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif t == "temporal_cnn":
        model = TemporalCNN(**model_kwargs)
    elif t == "st_vit":
        return SpatiotemporalViT(**model_kwargs)
    elif t == "unet2d":
        return UNet2D_TimeAsChannels(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {t}")

    return model


# --- Model Architectures ---

# ----------------------------------------------------------------------
# Utility blocks (re‑use across models)
# ----------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual 2‑D block identical to your original."""
    def __init__(self, n_input_channels, n_output_channels, k=3, s=1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input_channels, n_output_channels, k, stride=s, padding=k // 2)
        self.bn1   = nn.BatchNorm2d(n_output_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output_channels, n_output_channels, k, padding=k // 2)
        self.bn2   = nn.BatchNorm2d(n_output_channels)
        self.skip  = nn.Sequential()
        if s != 1 or n_input_channels != n_output_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(n_input_channels, n_output_channels, 1, stride=s), nn.BatchNorm2d(n_output_channels)
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
        reduced = max(1, c // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, reduced, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(reduced, c, 1, bias=False), nn.Sigmoid()
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
        self.hidden_dim = hid_c

    def init_state(self, batch_size: int, spatial_size: tuple[int, int], device=None):
        """Return h₀, c₀ each shaped (B, hidden_dim, H, W)."""
        H, W = spatial_size
        device = device or next(self.parameters()).device
        h0 = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c0 = torch.zeros_like(h0)
        return h0, c0

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
# 2. UNet2D_TImeAsChannels
# ----------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two consecutive 3×3 convolutions + BN + ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet2D_TimeAsChannels(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, base_features=64, depth=4):
        super().__init__()
        # Save for forward
        self.depth = depth

        # --------------------
        # Encoder path
        # --------------------
        self.encoders = nn.ModuleList()
        in_ch = n_input_channels
        out_ch = base_features
        # First encoder block
        self.encoders.append(DoubleConv(in_ch, out_ch))
        # Subsequent encoder blocks (each halves H×W, doubles channels)
        for _ in range(1, depth):
            in_ch = out_ch
            out_ch = out_ch * 2
            self.encoders.append(DoubleConv(in_ch, out_ch))

        # MaxPool layer to use between encoder blocks
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --------------------
        # Bottleneck (bottom of U)
        # --------------------
        in_ch = out_ch
        out_ch = out_ch * 2
        self.bottleneck = DoubleConv(in_ch, out_ch)

        # --------------------
        # Decoder path
        # --------------------
        self.uptrans = nn.ModuleList()
        self.decoders = nn.ModuleList()
        in_ch = out_ch
        for _ in range(depth):
            # ConvTranspose2d upsamples by factor of 2
            out_ch = in_ch // 2
            self.uptrans.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # After concatenation, channel count doubles, so DoubleConv(in_ch, out_ch)
            self.decoders.append(DoubleConv(in_ch, out_ch))
            in_ch = out_ch

        # --------------------
        # Final 1×1 conv to map to desired output channels
        # --------------------
        self.final_conv = nn.Conv2d(base_features, n_output_channels, kernel_size=1)

    def forward(self, x):
        """
        x: (B, n_input_channels, H, W), where n_input_channels = M × T
        returns: (B, n_output_channels, H, W)
        """
        skip_connections = []
        out = x

        # Encoder: collect skip connections
        for enc in self.encoders:
            out = enc(out)
            skip_connections.append(out)
            out = self.pool(out)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder: upsample + concatenate + double conv
        for idx in range(self.depth):
            out = self.uptrans[idx](out)
            skip = skip_connections[-(idx + 1)]

            # In case the upsampled size doesn't exactly match due to rounding, interpolate:
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)

            # Concatenate along channel dim
            out = torch.cat([skip, out], dim=1)
            out = self.decoders[idx](out)

        # Final 1×1 conv to produce output_channels
        out = self.final_conv(out)
        return out

# ----------------------------------------------------------------------
# 3. SpatiotemporalViT (our “st_vit” type)
# ----------------------------------------------------------------------
class SpatiotemporalViT(nn.Module):
    """
    A simple Vision Transformer that takes [B, T, 1, H, W] → [B, 1, H, W].
    Hidden_channels = d_model (embedding size).
    Depth = number of TransformerEncoder layers.
    n_heads = number of attention heads.
    patch_size = P (assumes H, W divisible by P).
    """
    def __init__(
        self,
        n_input_channels:  int = 1,
        n_output_channels: int = 1,
        hidden_channels:   int = 128,   # d_model
        depth:             int = 4,     # num Transformer layers
        n_heads:           int = 8,
        patch_size:        int = 8,
        dropout_rate:      float = 0.1,
        window_length:     int = 3,
        H:                 int = 48,
        W:                 int = 72,
    ):
        super().__init__()
        self.P = patch_size
        self.T = window_length
        self.d_model = hidden_channels
        self.n_heads = n_heads
        self.depth = depth

        self.n_patches_h = H // self.P
        self.n_patches_w = W // self.P
        self.num_patches = self.n_patches_h * self.n_patches_w

        # 1) Patch embedding: each P×P → d_model
        self.patch_embed = nn.Conv2d(n_input_channels, hidden_channels, kernel_size=self.P, stride=self.P)

        # 2) Spatial position embeddings (num_patches × d_model)
        self.spatial_pos = nn.Parameter(torch.randn(1, self.num_patches, hidden_channels))

        # 3) Temporal position embeddings (window_length × d_model)
        self.temporal_pos = nn.Parameter(torch.randn(1, self.T, hidden_channels))

        # 4) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=n_heads,
            dim_feedforward=4 * hidden_channels,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=False,  # PyTorch default: sequence first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 5) Linear decoder: d_model → (P×P)
        self.decoder_lin = nn.Linear(hidden_channels, self.P * self.P)

        # 6) Final conv to blend patches back to one channel
        self.final_conv = nn.Conv2d(1, n_output_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, T, 1, H, W]
        if x.ndim == 4:
            x = x.unsqueeze(1)  # allow [B, 1, H, W]
        B, T, C, H, W = x.shape
        assert T == self.T, f"Expected window length {self.T}, got {T}"

        # 1) Patchify each t
        tokens = []
        for t in range(T):
            patch_feats = self.patch_embed(x[:, t])  # [B, d_model, H/P, W/P]
            patch_feats = patch_feats.flatten(2).transpose(1, 2)
            # [B, num_patches, d_model]
            tokens.append(patch_feats + self.spatial_pos)

        tokens = torch.stack(tokens, dim=1)  # [B, T, num_patches, d_model]
        # Add temporal pos (broadcast over num_patches)
        tokens = tokens + self.temporal_pos.unsqueeze(2)  # [1, T, 1, d_model]

        # Merge T & num_patches into sequence dim S = T * num_patches
        tokens = tokens.view(B, T * self.num_patches, self.d_model)  # [B, S, d_model]
        tokens = tokens.permute(1, 0, 2)  # [S, B, d_model]

        # 2) Transformer
        out_tokens = self.transformer(tokens)  # [S, B, d_model]
        out_tokens = out_tokens.permute(1, 0, 2).view(B, T, self.num_patches, self.d_model)

        # 3) Take only the last time slice (predict T+1 from t=1..T)
        last_tokens = out_tokens[:, -1, :, :]  # [B, num_patches, d_model]
        decoded = self.decoder_lin(last_tokens)  # [B, num_patches, P×P]

        # 4) Reconstruct to [B, 1, H, W]
        decoded = decoded.view(B, self.n_patches_h, self.n_patches_w, self.P, self.P)
        decoded = decoded.permute(0, 1, 3, 2, 4).reshape(B, 1, H, W)
        return self.final_conv(decoded)  # [B, 1, H, W]