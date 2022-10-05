import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from src.ops.dwconv import dwconv2d_ref, dwconv2d_unfold, dwconv1d_fold, dwconv2d_fold, dwconv2d_fft


@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('bias', [False])
@pytest.mark.parametrize('dilation', [3])
@pytest.mark.parametrize('padding', [9])
@pytest.mark.parametrize('kernel_size', [7])
@pytest.mark.parametrize('input_size', [7, 14, 28, 56])
@pytest.mark.parametrize('channels', [320])
def test_dwconv2d_unfold(channels, input_size, kernel_size, padding, dilation, bias, dtype):
    device = 'cuda'
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    x = torch.randn(batch_size, channels, input_size, input_size, device=device,
                    dtype=dtype).to(memory_format=torch.channels_last)
    w = torch.randn(channels, 1, kernel_size, kernel_size, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype) if bias else None
    out_ref = dwconv2d_ref(x, w, bias, padding, dilation)
    out = dwconv2d_unfold(x, w, bias, padding, dilation)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('dilation', [2])
@pytest.mark.parametrize('padding', [2])
@pytest.mark.parametrize('kernel_size', [7])
@pytest.mark.parametrize('input_size', [196])
@pytest.mark.parametrize('channels', [320])
def test_dwconv1d_fold(channels, input_size, kernel_size, padding, dilation, bias, dtype):
    device = 'cuda'
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    x = torch.randn(batch_size, channels, input_size, device=device, dtype=dtype)
    w = torch.randn(channels, 1, kernel_size, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype) if bias else None
    out_ref = F.conv1d(x, w, bias, padding=padding, dilation=dilation, groups=w.shape[0])
    out = dwconv1d_fold(x, w, bias, padding, dilation)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('dilation', [3])
@pytest.mark.parametrize('padding', [9])
@pytest.mark.parametrize('kernel_size', [7])
@pytest.mark.parametrize('input_size', [7, 14, 28, 56])
@pytest.mark.parametrize('channels', [320])
def test_dwconv2d_fold(channels, input_size, kernel_size, padding, dilation, bias, dtype):
    device = 'cuda'
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    x = torch.randn(batch_size, channels, input_size, input_size, device=device,
                    dtype=dtype).to(memory_format=torch.channels_last)
    w = torch.randn(channels, 1, kernel_size, kernel_size, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype) if bias else None
    out_ref = dwconv2d_ref(x, w, bias, padding, dilation)
    out = dwconv2d_fold(x, w, bias, padding, dilation)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('bias', [False])
@pytest.mark.parametrize('dilation', [3])
@pytest.mark.parametrize('padding', [9])
@pytest.mark.parametrize('kernel_size', [7])
@pytest.mark.parametrize('input_size', [7, 14, 28, 56])
@pytest.mark.parametrize('channels', [320])
def test_dwconv2d_fft(channels, input_size, kernel_size, padding, dilation, bias, dtype):
    device = 'cuda'
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 32
    x = torch.randn(batch_size, channels, input_size, input_size, device=device,
                    dtype=dtype).to(memory_format=torch.channels_last)
    w = torch.randn(channels, 1, kernel_size, kernel_size, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype) if bias else None
    out_ref = dwconv2d_ref(x, w, bias, padding, dilation)
    out = dwconv2d_fft(x, w, bias, padding, dilation)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
