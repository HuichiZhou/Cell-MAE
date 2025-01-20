import logging
from typing import List

import torch
from torch import nn as nn
import torch.nn.functional as F

from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * in_chans
        self.flatten = flatten

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.projs = nn.ModuleList([
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
            for _ in range(in_chans)
        ])
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        embeddings= [] 
        for i, proj in enumerate(self.projs):
            patch_embeddings = proj(x[:, i:i+1, :, :])  # Extract channel i and apply its CNN
            patch_embeddings = patch_embeddings.flatten(2)  # Flatten spatial dimensions (H, W) -> Patches
            patch_embeddings = patch_embeddings.transpose(1, 2)  # BCH -> BPC
            embeddings.append(patch_embeddings)
        # x = self.proj(x)
        
        embeddings = torch.stack(embeddings, dim=1)
        
        embeddings = embeddings.flatten(start_dim=1, end_dim=2)
        
        # print(embeddings)
        # if self.flatten:
        #     x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(embeddings)
        return x