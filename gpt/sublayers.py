from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt.utils.gpt_config import GPTconfig

class SelfAttention(nn.Module):
    def __init__(self, config: GPTconfig) -> None:
        super(SelfAttention, self).__init__()

        self.embd_size = config.embd_size
        self.n_heads = config.n_heads

        assert self.embd_size % self.n_heads == 0, f"d_model({config.embd_size}) must be divisible by n_heads({config.n_heads})."

        self.key = nn.Linear(config.embd_size, config.embd_size)
        self.query = nn.Linear(config.embd_size, config.embd_size)
        self.value = nn.Linear(config.embd_size, config.embd_size)

        self.projection = nn.Linear(config.embd_size, config.embd_size)

        self.attn_drop = nn.Dropout(config.attn_drop)
        self.resid_drop = nn.Dropout(config.resid_drop)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))


    def forward(self, x) -> torch.Tensor:
        B,T,C = x.size()

        # shape of k,q,v would be (batch_size, n_heads, block_size, hs)
        k = self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        q = self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)

        # (batch_size, n_heads, block_size, hs) x (batch_size, n_heads, hs, block_size) -> (batch_size, n_heads, block_size, block_size)
        attn = (q @ k.transpose(2,3)) / sqrt(k.size(-1))
        attn = attn.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # (batch_size, n_heads, block_size, block_size) x (batch_size, n_heads, block_size, hs) -> (batch_size, n_heads, block_size, hs)
        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)

        y = self.projection(y)
        y = self.resid_drop(y)

        return y

class PointWiseFeedForward(nn.Module):

    def __init__(self, config: GPTconfig) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
                                nn.Linear(config.embd_size, config.embd_size * 4),
                                nn.GELU(),
                                nn.Linear(config.embd_size * 4, config.embd_size),
                                nn.Dropout(config.resid_drop)
                                )

    def forward(self, x):
        return self.mlp(x)