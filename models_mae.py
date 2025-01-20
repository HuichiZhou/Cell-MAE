from functools import partial
import math
import torch
import torch.nn as nn

from timm.models.vision_transformer import Block

from util.patch_embed import PatchEmbed
from util.pos_embed_3d import get_3d_sincos_pos_embed

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=6,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True) 

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
     
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int((self.patch_embed.num_patches/self.in_chans)**.5),self.in_chans, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_3d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int((self.patch_embed.num_patches/self.in_chans)**.5),self.in_chans, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        for proj in self.patch_embed.projs:
            w = proj.weight.data  
            
            w_flat = w.view([w.shape[0], -1]) 
            torch.nn.init.xavier_uniform_(w_flat)
            
            proj.weight.data.copy_(w_flat.view_as(w))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0] 
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0 

        h = w = imgs.shape[2] // p 

        x = imgs.reshape(shape=(imgs.shape[0], 6, h, p, w, p)) 
        x = torch.einsum('nchpwq->nchwpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w*6, p**2 )) 
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 6))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 6, h * p, h * p))
        return imgs

    
    def random_masking(self, x, mask_ratio):
        batch_size, num_patches_with_channels, feature_dim = x.shape
        num_patches = 196  
        num_channels = 6   

        x_reshaped = x.view(batch_size, num_patches, num_channels, feature_dim)

        total_channels = num_channels  
        num_to_mask_float = total_channels * mask_ratio
        num_to_mask_lower = int(num_to_mask_float) 
        num_to_mask_upper = num_to_mask_lower + 1  

        num_upper_samples = round((num_to_mask_float - num_to_mask_lower) * num_patches)
        num_lower_samples = num_patches - num_upper_samples

        rand_idx = torch.rand(batch_size, num_patches, num_channels, device=x.device).argsort(dim=-1)

        mask = torch.ones(batch_size, num_patches, num_channels, device=x.device)

        mask[:, :num_lower_samples, rand_idx[:, :num_lower_samples, :num_to_mask_lower]] = 1

        mask[:, num_lower_samples:, rand_idx[:, num_lower_samples:, :num_to_mask_upper]] = 1

        mask[:, :, 0] = 0  

        mask_expanded = mask.unsqueeze(-1)  
        x_masked = x_reshaped * (1 - mask_expanded)

        x_masked = x_masked.view(batch_size, num_patches_with_channels, feature_dim)
        mask = mask.view(batch_size, num_patches_with_channels) 

        ids_restore = torch.argsort(rand_idx, dim=-1).view(batch_size, num_patches_with_channels)

        return x_masked, mask, ids_restore
    def random_masking_per_patch_same_keep_count(self, x, mask_ratio=0.75):
        N, L, D = x.shape

   
        keep_float = (1 - mask_ratio) * 6  
        keep_floor = math.floor(keep_float) 
        keep_ceil  = math.ceil(keep_float)   
        fraction   = keep_float - keep_floor

        r = torch.rand(())
        if r < fraction:
            keep_count = keep_ceil
        else:
            keep_count = keep_floor

        keep_count = max(1, min(6, keep_count)) 

        noise = torch.rand(N, 196, 6, device=x.device)   
        sorted_idx = torch.argsort(noise, dim=-1)       

        ids_shuffle = torch.full((N, L), -1, dtype=torch.long, device=x.device)

        for n in range(N):
            offset = 0
            for i in range(196):
                topk_idx = sorted_idx[n, i, :keep_count]
                for c in topk_idx:
                    global_pos = i * 6 + c.item()
                    ids_shuffle[n, offset] = global_pos
                    offset += 1
            used_mask = torch.zeros(L, dtype=torch.bool, device=x.device)
            used_mask[ids_shuffle[n, :offset]] = True
            all_positions = torch.arange(L, device=x.device)
            masked_positions = all_positions[~used_mask] 
            ids_shuffle[n, offset:] = masked_positions

        ids_restore = torch.argsort(ids_shuffle, dim=1) 

        x_shuffled = x.gather(
            dim=1,
            index=ids_shuffle.unsqueeze(-1).expand(-1, -1, D)
        )
        x_masked = x_shuffled[:, : 196 * keep_count, :]

        mask_ones = torch.ones((N, L), device=x.device, dtype=torch.long)
        mask_ones[:, : 196 * keep_count] = 0
        mask = torch.gather(mask_ones, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
   

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        x, mask, ids_restore = self.random_masking_per_patch_same_keep_count(x, mask_ratio)
      
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  
        x = torch.cat([x[:, :1, :], x_], dim=1) 

        x = x + self.decoder_pos_embed
        
        

        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)


        x = self.decoder_pred(x)

        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
      
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
     
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  

        loss = (loss * mask).sum() / mask.sum()  
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore) 
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=528, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=516, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=516, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b 
