import torch
import torch.nn as nn
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    # raw vars
    # model_kwargs["n_input_channels"]  = len(cfg.data.input_vars)
    
    # raw vars, month encoding, spatial features (coarse), deltas and cumulatives
    # model_kwargs ["n_input_channels"] = (3 * len(cfg.data.input_vars) + 2) * 2

    # raw vars, month encoding, deltas and cumulatives
    model_kwargs["n_input_channels"] = 3 * len(cfg.data.input_vars) + 2
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)

    t = cfg.model.type
    if t == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif t == "temporal_cnn":
        model = TemporalCNN(**model_kwargs)
    elif t == "temporal_tcn":
        model = TemporalTCN(**model_kwargs)
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
# 6. SimVP-v2  (3-D Conv encoder–decoder with Gated ST Attention)
# ----------------------------------------------------------------------
class GSTA(nn.Module):
    """Gated Spatiotemporal Attention – single SE-style gate on 3-D feature."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c, c // r, 1), nn.ReLU(inplace=True),
            nn.Conv3d(c // r, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

def _conv3d_block(in_c, out_c, k=(3,3,3), s=1, p=(1,1,1)):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, k, stride=s, padding=p),
        nn.BatchNorm3d(out_c), nn.ReLU(inplace=True)
    )

class SimVPv2(nn.Module):
    r"""[B,T,C,H,W] → [B,C_out,H,W]"""
    def __init__(self, n_input_channels, n_output_channels,
                 width=64, depth=4, dropout_rate=0.1):
        super().__init__()
        self.enc = nn.Sequential(*[
            _conv3d_block(n_input_channels if i==0 else width,
                          width) for i in range(depth)
        ])
        self.gate = GSTA(width)
        self.dec = nn.Sequential(*[
            _conv3d_block(width, width) for _ in range(depth)
        ])
        self.head = nn.Conv2d(width, n_output_channels, 1)
        self.dp   = nn.Dropout(dropout_rate)

    def forward(self, x):
        # [B,T,C,H,W] → [B,C,T,H,W]
        x = x.permute(0,2,1,3,4).contiguous()
        z = self.gate(self.enc(x))
        z = self.dec(z)
        # take last timestep
        z = z[:, :, -1]          # [B,C,H,W]
        return self.head(self.dp(z))


# ----------------------------------------------------------------------
# 7. E3D-LSTM  (Eidetic 3-D ConvLSTM stack)
# ----------------------------------------------------------------------
class E3DLSTMCell(nn.Module):
    def __init__(self, in_c, hid_c, k=3):
        super().__init__()
        p = k//2
        self.conv = nn.Conv3d(in_c + hid_c, 7*hid_c,
                              kernel_size=(1,k,k),
                              padding=(0,p,p))

    def forward(self, x, h, c, m):
        # x,h,c,m : [B, C, 1, H, W]  (keep T dim =1)
        if m is None:
            m = torch.zeros_like(c)
        feats = torch.cat([x, h, m], dim=1)
        gates = self.conv(feats).chunk(7, dim=1)
        i,f,o,g,ii,ff,gg = gates
        i,f,o,ii,ff = map(torch.sigmoid, (i,f,o,ii,ff))
        g,gg = map(torch.tanh, (g,gg))
        c_next = f*c + i*g
        m_next = ff*m + ii*gg
        h_next = o*torch.tanh(c_next)
        return h_next, c_next, m_next

class E3DLSTM(nn.Module):
    def __init__(self, n_input_channels, n_output_channels,
                 hid_c=64, layers=2):
        super().__init__()
        self.spat_enc = nn.Conv3d(n_input_channels, hid_c, 3, padding=1)
        self.cells = nn.ModuleList([E3DLSTMCell(hid_c, hid_c)
                                    for _ in range(layers)])
        self.head = nn.Conv2d(hid_c, n_output_channels, 1)

    def forward(self, x):              # [B,T,C,H,W]
        B,T,C,H,W = x.shape
        h = [torch.zeros(B, 64, 1, H, W, device=x.device) for _ in self.cells]
        c = [torch.zeros_like(hh) for hh in h]
        m = None
        for t in range(T):
            z = self.spat_enc(x[:,t].unsqueeze(2))   # add dummy T dim
            h[0],c[0],m = self.cells[0](z,h[0],c[0],m)
            for l in range(1,len(self.cells)):
                h[l],c[l],m = self.cells[l](h[l-1],h[l],c[l],m)
        out = h[-1].squeeze(2)   # [B,C,H,W]
        return self.head(out)


# ----------------------------------------------------------------------
# 8. STCNet  (Hybrid temporal 1-D + spatial 2-D convs)
# ----------------------------------------------------------------------
class STCNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels,
                 t_channels=64, k_temporal=3, blocks=4):
        super().__init__()
        # 1-D temporal conv over channel-stacked map (depth-wise)
        self.temporal = nn.Sequential(*[
            nn.Conv3d(n_input_channels, n_input_channels,
                      kernel_size=(k_temporal,1,1),
                      padding=((k_temporal-1)//2,0,0),
                      groups=n_input_channels),
            nn.ReLU(inplace=True)
        ])
        # spatial encoder
        layers=[]; c=n_input_channels
        for _ in range(blocks):
            layers += [ResidualBlock(c, t_channels),]
            c = t_channels
        self.spatial = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, n_output_channels, 1)

    def forward(self, x):                 # [B,T,C,H,W]
        # fuse time with depth-wise temporal conv
        z = self.temporal(x.permute(0,2,1,3,4))      # [B,C,T,H,W]
        # collapse T by max-pool
        z = torch.max(z, dim=2).values               # [B,C,H,W]
        z = self.spatial(z)
        return self.head(z)

