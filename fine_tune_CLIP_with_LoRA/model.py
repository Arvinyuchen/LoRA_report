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
    

def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        assert d_model % n_head == 0, "Embedding dimension must be 0 modulo number of heads."

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.qkv_proj = LoRALinear(d_model, 3 * d_model)
        self.out_proj = LoRALinear(d_model, d_model)

        self.ln_1 = LayerNorm(d_model)

        self.mlp= nn.Sequential(OrderedDict([
            ("c_fc", LoRALinear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", LoRALinear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def attention(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.size()
        print(x.shape)

        if self.attn_mask is not None:
            self.attn_mask = expand_mask(self.attn_mask)
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.n_head, 3*self.d_head)
        qkv = qkv.permute(0, 2, 1, 3)   # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        def scaled_dot_product(q, k, v, mask=None):
            d_k = q.size()[-1]
            attn_logits = torch.matmul(q, k.transpose(-2, -1))
            attn_logits = attn_logits / math.sqrt(d_k)
            print(attn_logits.shape)

            if self.attn_mask is not None: #and self.attn_mask.shape[-1] == seq_len
                print(self.attn_mask.shape)
                attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
            attention = F.softmax(attn_logits, dim=-1)
            values = torch.matmul(attention, v)
            return values, attention
        
        values, attention = scaled_dot_product(q, k, v, mask=self.attn_mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_len , self.d_model)
        
        return self.out_proj(values)


    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
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
                attn_mask=block.attn_mask
            )

    # Patch the vision transformer
    patch_transformer(clip_model.visual.transformer)

    # Patch the text transformer
    patch_transformer(clip_model.transformer)

