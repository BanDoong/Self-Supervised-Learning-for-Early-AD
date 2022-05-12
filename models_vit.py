import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0., classification=False, conv=False):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)

        self.classification = classification
        self.conv = conv

        assert image_height % patch_height == 0 and image_width % patch_width == 0 and image_depth % patch_height == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # nn.Conv3d(in_channels=channels, out_channels=patch_dim,kernel_size=patch_size, stride=patch_size)
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=patch_height, p2=patch_width,
                      p3=patch_depth),
            nn.Linear(patch_dim, dim),
        )
        self.dim = dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        import numpy as np
        self.blocks = np.arange(depth)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        if self.conv:
            # 256x256x256
            self.convolutions_1 = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(num_features=8),
                nn.GELU())
            # 128x128x128
            self.convolutions_2 = nn.Sequential(
                nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(num_features=16),
                nn.GELU(), )
            # 64x64x64
            self.convolutions_3 = nn.Sequential(
                nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(num_features=32),
                nn.GELU(), )
            # 32x32x32
            self.convolutions_4 = nn.Sequential(
                nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(num_features=32),
                nn.GELU(), )
            # # 16x16x16
            # self.convolutions_5 = nn.Sequential(
            #     nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            #     nn.BatchNorm3d(num_features=32),
            #     nn.GELU(), )

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, img):
        if self.conv:
            img = self.convolutions_1(img)
            img = self.convolutions_2(img)
            img = self.convolutions_3(img)
            img = self.convolutions_4(img)
            # img = self.convolutions_5(img)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


def vit_base_patch16(**kwargs):
    model = ViT(image_size=128, patch_size=16, dim=768, depth=12, heads=12, num_classes=2, mlp_dim=1024)
    return model


def vit_large_patch16(**kwargs):
    model = ViT(image_size=128, patch_size=16, dim=1024, depth=24, heads=16, num_classes=2, mlp_dim=1024)
    return model


def vit_huge_patch14(**kwargs):
    model = ViT(image_size=128, patch_size=8, dim=1280, depth=32, heads=16, num_classes=2, mlp_dim=1024)
    return model


def conv_vit_base_patch16(**kwargs):
    model = ViT(image_size=16, patch_size=8, dim=1280, depth=32, heads=16, num_classes=2, mlp_dim=1024, channels=32,
                conv=True)
    return model


def resize_vit_base(**kwargs):
    model = ViT(image_size=(121, 144, 121), patch_size=(11, 12, 11), dim=768, depth=12, heads=12, num_classes=2,
                mlp_dim=1024)
    return model


vit_base_patch16 = vit_base_patch16
vit_large_patch16 = vit_large_patch16
vit_huge_patch14 = vit_huge_patch14
conv_vit_base_patch16 = conv_vit_base_patch16
resize_vit_base = resize_vit_base

# from torchinfo import summary
#
# model = resize_vit_base()
# summary(model, (4, 1, 121, 144, 121))
