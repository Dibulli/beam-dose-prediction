from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import numpy as np
import h5py as h5
import einops



# helpers
"""Segformer with GAN."""

def exists(val):
    return val is not None


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class EfficientSelfAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads,
            reduction_ratio
    ):
        #   reduction_ratio ---> decide how many times the size of k & v is different from q (or original data).
        #   kernel size ---> used with dilation
        #   the OutPut size of vector after Conv2d only rely on stride

        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)


    def forward(self, x):
        h, w = x.shape[-2:]
        # height and weight of matrix = [batch, dim, height, weight]

        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h=heads, x=h, y=w)
        return self.to_out(out)


class MixFeedForward(nn.Module):
    def __init__(
            self,
            *,
            dim,
            expansion_factor
    ):
        # dim of matrix = [batch, dim, height, weight]
        # expansion_factor ---> 
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        # return self.net(x)
        return nn.functional.softmax(self.net(x), dim=0)


class MiT(nn.Module):
    def __init__(
            self,
            *,
            channels,
            dims,
            heads,
            ff_expansion,
            reduction_ratio,
            num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))
        # dim_pairs: [(16, 32), (32, 64), (64, 128), (128, 256)]
        # kernel --- [7,        3,        3,         3]
        # stride --- [4,        2,        2,         2]
        # padding -- [3,        1,        1,         1]
        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(
                dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride=stride, padding=padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim=dim_out, heads=heads, reduction_ratio=reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim=dim_out, expansion_factor=ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
            self,
            x,
            return_layer_outputs=False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h=h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            layer_outputs.append(x)

        ret = x if not return_layer_outputs else layer_outputs
        return ret


class Segformer(nn.Module):
    def __init__(
            self,
            *,
            dims=(32, 64, 160, 256),
            # dims=(64, 128, 256, 512),
            # heads=(1, 2, 5, 8),
            heads = (1, 2, 4, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            num_layers=2,
            channels=6,
            # channels=24,

            # mask_channel=14,
            # channels = number of masks + 1(CT image) + 1 (Beam contour channel)
            # channels = 3,
            decoder_dim=256,
            # decoder_dim=512,

            num_classes=1
    ):
        # ff_expansion ---> feed forward layer expansion, dims * ff_expansion = hidden dims.

        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth=4), (
        dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio,
                                                 num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels=channels,
            dims=dims,
            heads=heads,
            ff_expansion=ff_expansion,
            reduction_ratio=reduction_ratio,
            num_layers=num_layers
        )
        # self.mask_mit = MiT(
        #     channels=mask_channel,
        #     dims=dims,
        #     heads=heads,
        #     ff_expansion=ff_expansion,
        #     reduction_ratio=reduction_ratio,
        #     num_layers=num_layers
        # )
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor=2 ** i, mode = 'bilinear')
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(decoder_dim, num_classes, 1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            # nn.Sigmoid()
        )

    def forward(self, x):
        layer_outputs = self.mit(x, return_layer_outputs=True)
        # mask_outputs = self.mask_mit(masks, return_layer_outputs=False)

        # fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        # layer_outputs[-1] = layer_outputs[-1] + mask_outputs
        # fused = torch.cat(fused, dim=1)
        fused=[]
        for output, to_fused in zip(layer_outputs, self.to_fused):
            fused.append(to_fused(output))
        fused = torch.cat(fused, dim=1)
        return self.to_segmentation(fused)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 24, normalization=False),
            *discriminator_block(24, 48),
            *discriminator_block(48, 96),
            *discriminator_block(96, 192),
            # *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            # nn.Conv2d(512, 1, 4, padding=1, bias=False)
            nn.Conv2d(192, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A):
        # Concatenate image and condition image by channels to produce input
        # img_input = torch.cat((img_A, img_B), 1)
        # return self.model(img_input)
        return self.model(img_A)

if __name__ == "__main__":
    print("debug")
    file = "/Volumes/NPC预测数据盘/gan256_no_gan/test_h5_256_86slices/e9278201.h5"
    p = h5.File(file)
    d = p["slice_dose"]
    m = p["slice_mask"]
    im = p["slice_img"]
    b = p["slice_beam"]
    beam_slice = b[:, :, :, 30, 0]
    bs = rearrange(beam_slice, '(b c) h w -> b c h w', b=1)

    model = Segformer(channels=5)
    pred = model(torch.from_numpy(bs))
    print("d")
    print(pred.shape)
