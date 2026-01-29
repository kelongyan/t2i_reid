# models/vim.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class PatchEmbed(nn.Module):
    # 2D 图像到 Patch 嵌入层
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
    # 无偏置项的 LayerNorm，用于适配特定预训练权重
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
        if self.elementwise_affine:
            self.register_parameter('bias', None)

class VimMamba(nn.Module):
    # Vision Mamba 核心模块：实现双向 SSM（状态空间模型）处理
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

        # 1. 输入投影（前向与反向共享）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 2. 正向扫描分支组件
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM 正向参数 A 和 D
        A = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 3. 反向扫描分支组件
        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=conv_bias,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM 反向参数 A 和 D
        A_b = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner).contiguous()
        self.A_b_log = nn.Parameter(torch.log(A_b))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))

        # 4. 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):
        # hidden_states: [B, L, D]
        batch, seqlen, dim = hidden_states.shape

        # 输入 NaN 容错处理
        if torch.isnan(hidden_states).any():
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)

        # 投影并切分为 x 和门控 z
        xz = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)

        # === 正向路径处理 ===
        x_fwd = self.conv1d(x)[:, :, :seqlen] if self.d_conv > 1 else x
        x_fwd = x_fwd.transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x_fwd)
        dt, B_fwd, C_fwd = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        # 数值范围裁剪，增强稳定性
        x_fwd = torch.clamp(x_fwd, min=-10.0, max=10.0)
        dt = torch.clamp(dt, min=1e-5, max=1.0)

        # 统一转换为 float32 进行 selective_scan 运算
        y_fwd = selective_scan_fn(
            x_fwd.transpose(1, 2).float(), dt.transpose(1, 2).float(), A.float(),
            B_fwd.transpose(1, 2).float(), C_fwd.transpose(1, 2).float(), self.D.float(),
            z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True, return_last_state=False
        )

        if torch.isnan(y_fwd).any():
            y_fwd = torch.nan_to_num(y_fwd, nan=0.0, posinf=1.0, neginf=-1.0)

        # === 反向路径处理 ===
        x_bwd = x.flip([-1])
        x_bwd = self.conv1d_b(x_bwd)[:, :, :seqlen] if self.d_conv > 1 else x_bwd
        x_bwd = x_bwd.transpose(1, 2)

        A_b = -torch.exp(self.A_b_log.float())
        x_dbl_b = self.x_proj_b(x_bwd)
        dt_b, B_bwd, C_bwd = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = self.dt_proj_b(dt_b)

        x_bwd = torch.clamp(x_bwd, min=-10.0, max=10.0)
        dt_b = torch.clamp(dt_b, min=1e-5, max=1.0)

        y_bwd = selective_scan_fn(
            x_bwd.transpose(1, 2).float(), dt_b.transpose(1, 2).float(), A_b.float(),
            B_bwd.transpose(1, 2).float(), C_bwd.transpose(1, 2).float(), self.D_b.float(),
            z=None, delta_bias=self.dt_proj_b.bias.float(), delta_softplus=True, return_last_state=False
        )

        if torch.isnan(y_bwd).any():
            y_bwd = torch.nan_to_num(y_bwd, nan=0.0, posinf=1.0, neginf=-1.0)

        # 还原反向序列并合并
        y_bwd = y_bwd.flip([2])
        y = (y_fwd + y_bwd) * F.silu(z)

        if torch.isnan(y).any():
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        out = self.out_proj(y.transpose(1, 2))
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


class VimBlock(nn.Module):
    # Vision Mamba 基础块：包含归一化层和双向 Mamba 混合层
    def __init__(self, dim, mixer_cls=VimMamba, norm_cls=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(d_model=dim, **kwargs)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class VisionMamba(nn.Module):
    # Vision Mamba (Vim) 主模型：采用 Mid-Cls-Token 策略处理图像序列
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=24, 
                 rms_norm=False, drop_path_rate=0., **kwargs):
        super().__init__()
        
        norm_layer = LayerNormNoBias if not rms_norm else nn.RMSNorm
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 堆叠 Mamba 块
        self.layers = nn.ModuleList([
            VimBlock(dim=embed_dim, mixer_cls=VimMamba, norm_cls=norm_layer)
            for _ in range(depth)
        ])
        
        self.norm_f = norm_layer(embed_dim)
        self.token_position = num_patches // 2

    def forward(self, x):
        # 1. 图像分块与嵌入
        x = self.patch_embed(x)
        B, N, C = x.shape
        
        # 2. 插入 Mid-Cls Token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((x[:, :self.token_position, :], cls_token, x[:, self.token_position:, :]), dim=1)
        
        # 3. 添加位置编码并逐层处理
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
            
        # 4. 最终归一化
        return self.norm_f(x)
