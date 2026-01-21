import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
            
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class LayerNormNoBias(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        if self.elementwise_affine:
            self.register_parameter('bias', None)

class VimMamba(nn.Module):
    """
    Vision Mamba Block with Bidirectional SSM.
    Matches the parameter structure of 'vim_s_midclstok.pth'.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path

        # 1. Input Projection (Shared for forward/backward)
        # Projects to [d_inner * 2] (splitting into x and z gates later)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 2. Forward Branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM Parameters (Forward)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 3. Backward Branch (Vision Mamba Specific)
        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM Parameters (Backward)
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)
        self.A_b_log = nn.Parameter(A_b_log)
        self.D_b = nn.Parameter(torch.ones(self.d_inner))

        # 4. Output Projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):
        """
        hidden_states: [B, L, D]
        """
        batch, seqlen, dim = hidden_states.shape

        # 1. Project and Split
        xz = self.in_proj(hidden_states).transpose(1, 2) # [B, 2*d_inner, L]
        x, z = xz.chunk(2, dim=1)

        # 2. Forward Path
        # Conv1d
        x_fwd = x
        if self.d_conv > 1:
            x_fwd = self.conv1d(x)[:, :, :seqlen]
        
        x_fwd = x_fwd.transpose(1, 2) # [B, L, d_inner]
        
        # SSM Forward
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x_fwd) # [B, L, dt_rank + 2*d_state]
        dt, B_fwd, C_fwd = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt) # [B, L, d_inner]
        
        # [Fix] Force ALL inputs to float32 for selective_scan_fn numerical stability & type matching
        x_fwd = x_fwd.float()
        dt = dt.float()
        A = A.float()
        B_fwd = B_fwd.transpose(1, 2).float() # [B, N, L]
        C_fwd = C_fwd.transpose(1, 2).float() # [B, N, L]
        D_fwd = self.D.float()
        delta_bias_fwd = self.dt_proj.bias.float()
        
        # selective_scan_fn requires [B, D, L] for u and delta
        y_fwd = selective_scan_fn(
            x_fwd.transpose(1, 2), # [B, D, L]
            dt.transpose(1, 2),    # [B, D, L]
            A,
            B_fwd, 
            C_fwd, 
            D_fwd,
            z=None, delta_bias=delta_bias_fwd, delta_softplus=True, return_last_state=False
        )
        
        # 3. Backward Path
        x_bwd = x.flip([-1]) # Reverse along sequence dimension
        if self.d_conv > 1:
            x_bwd = self.conv1d_b(x_bwd)[:, :, :seqlen]
            
        x_bwd = x_bwd.transpose(1, 2) # [B, L, d_inner]
        
        # SSM Backward
        A_b = -torch.exp(self.A_b_log.float())
        x_dbl_b = self.x_proj_b(x_bwd)
        dt_b, B_bwd, C_bwd = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = self.dt_proj_b(dt_b)
        
        # [Fix] Force ALL inputs to float32 for backward branch
        x_bwd = x_bwd.float()
        dt_b = dt_b.float()
        A_b = A_b.float()
        B_bwd = B_bwd.transpose(1, 2).float()
        C_bwd = C_bwd.transpose(1, 2).float()
        D_bwd = self.D_b.float()
        delta_bias_bwd = self.dt_proj_b.bias.float()
        
        y_bwd = selective_scan_fn(
            x_bwd.transpose(1, 2), # [B, D, L]
            dt_b.transpose(1, 2),  # [B, D, L]
            A_b,
            B_bwd,
            C_bwd,
            D_bwd,
            z=None, delta_bias=delta_bias_bwd, delta_softplus=True, return_last_state=False
        )
        
        # Flip backward output back
        y_bwd = y_bwd.flip([2]) # Flip along length dim (last dim)
        
        # 4. Gate and Combine
        y = y_fwd + y_bwd
        y = y * F.silu(z) # z is already [B, D, L]
        
        # 5. Output Project
        out = self.out_proj(y.transpose(1, 2))
        return out


class VimBlock(nn.Module):
    def __init__(self, dim, mixer_cls=VimMamba, norm_cls=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(d_model=dim, **kwargs)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class VisionMamba(nn.Module):
    """
    Vision Mamba (Vim-S) adapted for T2I-ReID.
    Support Mid-Cls-Token strategy.
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 embed_dim=384, 
                 depth=24, 
                 rms_norm=False, 
                 drop_path_rate=0.,
                 **kwargs):
        super().__init__()
        
        # Use LayerNormNoBias to avoid missing key warnings if checkpoint has no bias
        norm_layer = LayerNormNoBias if not rms_norm else nn.RMSNorm
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Blocks
        self.layers = nn.ModuleList([
            VimBlock(dim=embed_dim, mixer_cls=VimMamba, norm_cls=norm_layer)
            for _ in range(depth)
        ])
        
        # Final Norm
        self.norm_f = norm_layer(embed_dim)
        
        self.token_position = num_patches // 2  # Mid-Cls-Token Position

    def forward(self, x):
        # 1. Patch Embed
        x = self.patch_embed(x) # [B, 196, 384]
        B, N, C = x.shape
        
        # 2. Mid-Cls-Token Construction
        # Expand cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        
        # Split and Insert (Mid Strategy)
        # Note: pos_embed is [1, 197, 384], matches the concatenated sequence length
        x = torch.cat((x[:, :self.token_position, :], cls_token, x[:, self.token_position:, :]), dim=1)
        
        # Add Position Embedding
        x = x + self.pos_embed
        
        # 3. Process Layers
        for layer in self.layers:
            x = layer(x)
            
        # 4. Final Norm
        x = self.norm_f(x)
        
        # Return all tokens (let the downstream model decide how to use them, 
        # usually GS3 or GAP will handle the sequence)
        return x
