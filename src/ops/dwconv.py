# Different implementations of depthwise conv, targeting dilated dw conv for VAN.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange


def dwconv2d_ref(x, w, bias=None, padding=0, dilation=1):
    """
    x: (batch, channels, height, width)
    w: (channels, 1, kernel_height, kernel_width)
    output: (batch, channels, height', width')
    """
    return F.conv2d(x, w, bias, padding=padding, dilation=dilation, groups=w.shape[0])


def dwconv2d_unfold(x, w, bias=None, padding=0, dilation=1):
    """Unfolding is slower because it needs to materialize x_unf, which is many times larger
    than x.
    Assumes that the output shape is the same as input shape.
    """
    kernel_size = w.shape[-2:]
    # TD [2022-06-18] Pytorch unfold (im2col) loops over the batch dimension and executes @b kernels.
    # [Idk why]. Instead we fold the batch into the channel dimension to reduce it to 1 kernel launch.
    x_unf = rearrange(torch.nn.functional.unfold(rearrange(x, 'b c h w -> 1 (b c) h w'),
                                                 kernel_size, padding=padding, dilation=dilation),
                      '1 (b c k) hw -> b c k hw', b=x.shape[0], c=x.shape[1])
    w_unf = rearrange(w, 'c 1 kh kw -> c (kh kw)')
    out = rearrange(torch.einsum('bckz,ck->bcz', x_unf, w_unf), 'b c (h w) -> b c h w', w=x.shape[-1])
    return out + rearrange(bias, 'c -> c 1 1') if bias is not None else out


def dwconv1d_fold(x, w, bias=None, padding=0, dilation=1):
    """Assumes that output shape is the same as input shape
    """
    # Force NWC format
    x_folded = rearrange(rearrange(F.pad(x, (padding, padding)), 'b c (w d) -> (b d) w c', d=dilation),
                         'b w c -> b c w')
    out = rearrange(F.conv1d(x_folded, w, bias, groups=w.shape[0]), '(b d) c w -> b c (w d)',
                    b=x.shape[0])
    return out


class SubsetHW(torch.autograd.Function):
    """We just want input[:, :-padding, :-padding, :] but the backward pass of slicing is slower
    that it should be (it goes through 2 SliceBackward functions instead of just 1).
    https://github.com/pytorch/pytorch/issues/63921
    So we write a custom backward function.
    """

    @staticmethod
    def forward(ctx, input, padding):
        assert input.ndim == 4
        ctx.padding = padding
        return input[:, :-padding, :-padding, :]

    @staticmethod
    def backward(ctx, grad_output):
        padding = ctx.padding
        return F.pad(grad_output, (0, 0, 0, padding, 0, padding)), None


def dwconv2d_fold(x, w, bias=None, padding=9, dilation=3):
    """Assumes that the output shape is the same as the input shape
    """
    # I've only tested with these parameters
    assert padding == 9
    assert dilation == 3
    assert x.shape[-1] == x.shape[-2]
    assert w.shape[-1] == w.shape[-2]
    x_pad = ((x.shape[-1] + dilation - 1) // dilation) * dilation - x.shape[-1]
    x_padded = F.pad(x, (0, x_pad, 0, x_pad))
    x_folded = rearrange(x_padded, 'b c (h dh) (w dw) -> (b dh dw) h w c',
                         dw=dilation, dh=dilation)
    x_folded = rearrange(x_folded, 'b h w c -> b c h w')
    # TODO: What should padding be here in the general case?
    out = rearrange(F.conv2d(x_folded, w, bias, padding=3, groups=w.shape[0]), '(b dh dw) c h w -> b (h dh) (w dw) c', dh=dilation, dw=dilation)
    out = SubsetHW.apply(out, x_pad)
    return rearrange(out, 'b h w c -> b c h w')


def dwconv2d_fft(x, w, bias=None, padding=0, dilation=3):
    """The package fft_conv_pytorch implements fft_conv, but it's very slow for depthwise conv.
    We assume that the output shape is the same as the input shape.
    """
    assert dilation == 3
    w_long = rearrange(F.pad(w, (0, 0, 0, 0, 0, 8)), 'c (dh dw) h w -> c 1 (h dh) (w dw)',
                       dh=dilation, dw=dilation)[:, :, :-2, :-2]
    x_padded = F.pad(x, (padding, padding, padding, padding))
    L = (x_padded.shape[-2], x_padded.shape[-1])
    x_f = torch.fft.rfft2(x_padded.float(), s=L)
    # w_f = torch.fft.rfft2(w.float() / (L[0] * L[1]), s=L)
    w_f = torch.fft.rfft2(w_long.float(), s=L).conj()
    y_f = x_f * rearrange(w_f, 'c 1 h w -> 1 c h w')
    # out = torch.fft.irfft2(y_f, s=L, norm='forward')[:, :, :x.shape[-2], :x.shape[-1]]
    out = torch.fft.irfft2(y_f, s=L).to(dtype=x.dtype)[:, :, :x.shape[-2], :x.shape[-1]]
    return out + rearrange(bias, 'c -> c 1 1') if bias is not None else out
