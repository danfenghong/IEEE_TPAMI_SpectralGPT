# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


# import util.logging as logging
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool3d(1)  # 在深度维度上应用全局平均池化
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, num_channels, depth, height, width = x.size()

        # 在深度维度上进行全局平均池化
        out = self.global_avg_pooling(x)

        # 使用1x1卷积层计算通道权重
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))

        # 将权重应用到深度维度上
        out = out.view(batch_size, num_channels, 1, 1, 1)
        out = x * out

        return out


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        # temporal related:
        frames=32,
        t_patch_size=4,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        print(
            f"img_size {img_size} patch_size {patch_size} frames {frames} t_patch_size {t_patch_size}"
        )
        self.img_size = img_size
        self.patch_size = patch_size

        self.frames = frames
        self.t_patch_size = t_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]  # 12
        self.t_grid_size = frames // t_patch_size  # 3

        kernel_size = [t_patch_size] + list(patch_size)  # 3,8,8

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, x):

        B, C, T, H, W = x.shape  # 1,1,12,120,120


        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        assert T == self.frames

        x = self.proj(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        # print(q.shape)
        # print(v.shape)
        # print((k.transpose(-2, -1)).shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x

class Linear_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        # print(q.shape)
        # print(v.shape)
        # print((k.transpose(-2, -1) * self.scale).shape)

        # attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = ((k.transpose(-2, -1) * self.scale).softmax(dim=-1)) @ v

        # attn = attn.softmax(dim=-1)

        x = ((q.softmax(dim=-1)) @ attn).reshape(B, N, C)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x
class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# if __name__ == '__main__':
    # input = torch.rand(4,12,120,120)
    # input = torch.unsqueeze(input,dim=1)   #torch.Size([2, 1, 10, 512, 512]) B,T,C,H,W
    # patch_embed = PatchEmbed(img_size=120,in_chans=1,frames=12,t_patch_size=3,patch_size=8)
    # output = patch_embed(input)
if __name__ == '__main__':
        # from lovas_loss import lovasz_hinge,symmetric_lovasz
        #
        # out = torch.rand(4,96,96)
        # labels = torch.rand(4,96,96)
        #
        # # loss = lovasz_hinge(out, labels)*0.1
        # loss =symmetric_lovasz(out, labels)
        # loss_bce = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(1))
        # loss2 = loss_bce(out, labels)
        # print(loss)
        # print(loss2)


            # input = torch.rand(2,9,512,512)
            # input = torch.unsqueeze(input,dim=1)   #torch.Size([2, 1, 10, 512, 512]) B,T,C,H,W
            # patch_embed = PatchEmbed(img_size=512,in_chans=1,frames=9,t_patch_size=3)
            # output = patch_embed(input)
            x = torch.rand(2, 196, 768)
            model = Linear_Attention(dim=768)
            # model = Attention(dim=768)
            output = model(x)
            # print(output.shape)
