# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import xformers
import xformers.ops
from xformers.components.feedforward import MLP
from timm.models.vision_transformer import Attention as Attention
from xformers.components import Activation
from inspect import isfunction
from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
 
class PointEmbed(nn.Module):
    """Point Embedding (keep the original argument list for compatibility)
    """
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv1d(in_chans, embed_dim, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Point embedding
        B, Cin, N = x.shape

        # Point embedding
        point_emb = self.proj(x)
        
        return self.relu(point_emb)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        return self.mlp(t_freq)

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype
    
class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs//s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)

class ConditionEmbedder(nn.Module):
    """
    Embeds conditions into fourier features.
    """
    def __init__(self, hidden_size, uncond_prob, num_conditions, num_cyclic_conditions, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.num_cyclic_conditions = num_cyclic_conditions
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(num_conditions, hidden_size) / hidden_size ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, condition, force_drop_ids=False):
        """
        Drops labels to enable classifier-free guidance.
        """
        if not force_drop_ids:
            drop_ids = torch.rand(condition.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = torch.ones(condition.shape[0]).bool().cuda()
        condition = torch.where(drop_ids[:, None, None], self.y_embedding, condition)
        return condition

    @staticmethod
    def cyclic_embedding(condition, dim):
        batch_size, _ = condition.shape
        half_dim = dim // 2

        frequencies = (- torch.arange(0, half_dim) * np.log(10000) / (half_dim - 1)).exp()
        frequencies = frequencies[None, None, :].repeat(batch_size, 1, 1).cuda()

        sin_sin_emb = ((condition[:, :, None]).sin() * frequencies).sin()
        sin_cos_emb = ((condition[:, :, None]).cos() * frequencies).sin()
        emb = torch.cat([sin_sin_emb, sin_cos_emb], dim=2)
        
        if dim % 2:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb
    
    @staticmethod
    def positional_embedding(condition, dim):
        half_dim = dim // 2
        emb = np.ones(condition.shape[0]) * np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb[:,None])).float().to(torch.device('cuda'))
        emb = condition[:, :, None] * emb[:, None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=2)
        if dim % 2:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb

    def forward(self, c, train, force_drop_ids=None):
        c_freq_cyc = self.cyclic_embedding(c[:, :self.num_cyclic_conditions], self.frequency_embedding_size)
        c_freq_pos = self.positional_embedding(c[:, self.num_cyclic_conditions:], self.frequency_embedding_size)
        c_freq = torch.concat((c_freq_cyc, c_freq_pos), 1)
        c_emb = self.mlp(c_freq)
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids):
            c_emb = self.token_drop(c_emb, force_drop_ids)
        return c_emb
    
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **block_kwargs):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, context_dim=None, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)        
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.mlp = MLP(dim_model=hidden_size, hidden_layer_multiplier=int(mlp_ratio),
                       activation=Activation("gelu"), dropout=0)
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, c, context=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x, c):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone and clip encoder.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=1152,
        context_dim=768,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        num_classes=1,
        compile_components=False,
        num_cyclic_conditions=1,
        num_total_conditions=6,
        uncond_prob=.1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads

        self.x_embedder = PointEmbed(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, uncond_prob)

        self.final_layer = FinalLayer(hidden_size, self.out_channels)

        self.c_embedder = ConditionEmbedder(hidden_size, uncond_prob, num_conditions=num_total_conditions, num_cyclic_conditions=num_cyclic_conditions)

        self.secondary_device = torch.device("cpu")

        self.blocks = [
            DiTBlock(hidden_size, num_heads, context_dim=context_dim, mlp_ratio=mlp_ratio) for _ in range(depth)
        ]

        self.initialize_weights()
        self.blocks = nn.ModuleList(self.blocks)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize condition embedding MLP:
        nn.init.normal_(self.c_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.c_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
            # nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            # nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, c, force_dropout=False):
        """
        Forward pass of DiT3D.
        x: (B, C, N) tensor of point inputs
        t: (N,) tensor of diffusion timesteps
        context: (N, context_length, context_dim) embedding context
        """
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        # y = self.y_embedder(y)
        c = self.c_embedder(c, self.training, force_drop_ids=force_dropout)

        context = torch.cat((t[:, None, :], c), dim=1)
        # context = c

        # y = self.
        x = x.transpose(-1,-2)
        for block in self.blocks:
            x = block(x, t, context)  # (N, T, D)

        # left context in, but it's not used atm
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        del t

        return x.transpose(-1,-2)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XS_4(pretrained=False, **kwargs):

    model = DiT(depth=12, hidden_size=192, context_dim=192, patch_size=4, num_heads=3, **kwargs)
    if pretrained:
        checkpoint = torch.load('/home/ekirby/workspace/DiT-3D/checkpoints/shapenet_s4_scaled_1/last.ckpt', map_location='cpu')
        if "ema" in checkpoint:  # supports ema checkpoints 
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {k: checkpoint[k] for k in checkpoint if k.startswith('blocks')}
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model

DiT3D_models_JustCrossAttn = {
    'DiT-XS/4':  DiT_XS_4,
}