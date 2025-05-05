from collections import OrderedDict

import clip
import torch
import torch.nn as nn
import loralib.layers as lora
import math
from loralib.layers import MergedLinear, Linear as LoRALinear
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, r=4, lora_alpha=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        # Inject LoRA into in_proj_weight via MergedLinear
        self.attn_lora = MergedLinear(
            in_features=d_model,
            out_features=3 * d_model,  # qkv merged
            r=r,
            lora_alpha=lora_alpha,
            enable_lora=[True, True, True],  # enable for q, k, v
            fan_in_fan_out=False
        )

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4, r=r, lora_alpha=lora_alpha, enable_lora=[True], fan_in_fan_out=False)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model, r=r, lora_alpha=lora_alpha, enable_lora=[True], fan_in_fan_out=False)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        x_norm = self.ln_1(x)

        # Compute QKV + LoRA delta in one shot
        qkv = F.linear(x_norm, self.attn.in_proj_weight, self.attn.in_proj_bias) + self.attn_lora(x_norm)

        # Split qkv for attention call
        E = self.attn.embed_dim
        q, k, v = qkv.split(E, dim=-1)

        out, _ = self.attn(
            query=q, key=k, value=v,
            need_weights=False,
            attn_mask=self.attn_mask
        )
        return out

    def forward(self, x: torch.Tensor):
        x = x + self.attention(x)
        x = x + self.mlp(self.ln_2(x))
        return x
    


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


    
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x



def inject_lora_into_clip(clip_model, r=4, alpha=8):
    def patch_transformer(transformer: nn.Module):
        for i, block in enumerate(transformer.resblocks):
            transformer.resblocks[i] = ResidualAttentionBlock(
                d_model=block.attn.embed_dim,
                n_head=block.attn.num_heads,
                attn_mask=block.attn_mask,
                r=r,
                lora_alpha=alpha
            )

    # Patch the vision transformer
    patch_transformer(clip_model.visual.transformer)

    # Patch the text transformer
    patch_transformer(clip_model.transformer)

